"""Inference for Train internlm models."""
import gc
from typing import Iterable, Optional, Dict

import torch

from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length

class InferenceParams:
    """
    Infer parames.
    """

    def __init__(
        self,
        max_sequence_len,
        max_batch_size,
        sequence_len_offset=0,
        batch_size_offset=0,
        key_value_memory_dict: dict = None,
        lengths_per_sample=None,
        attention_mask=None,
    ) -> None:
        """
        推理相关的中间 cache 对象

        :param max_sequence_len: 最大长度
        :param max_batch_size: batch_size
        :param sequence_len_offset: _description_, defaults to 0
        :param batch_size_offset: _description_, defaults to 0
        :param key_value_memory_dict: _description_, defaults to None
        :param lengths_per_sample: _description_, defaults to None
        """
        self.max_sequence_len: int = max_sequence_len
        self.max_batch_size: int = max_batch_size
        self.sequence_len_offset: int = sequence_len_offset
        self.batch_size_offset: int = batch_size_offset
        if key_value_memory_dict is None:
            key_value_memory_dict = {}
        self.key_value_memory_dict: dict = key_value_memory_dict
        self.fused_ft_kernel: bool = False
        self.lengths_per_sample = lengths_per_sample  # 在 fused_ft_kernel 为 True 时会用到
        self.attention_mask = attention_mask

    @property
    def max_seqlen(self):
        return self.max_sequence_len

    @property
    def seqlen_offset(self):
        return self.sequence_len_offset

    def reorder_state(self, indices):
        # 在 beam search 期间会会涉及到重排的操作
        if self.lengths_per_sample is not None:
            self.lengths_per_sample = self.lengths_per_sample.index_select(index=indices, dim=0)
        for key, value in list(self.key_value_memory_dict.items()):
            value = value.index_select(index=indices, dim=0)
            self.key_value_memory_dict[key] = value

@torch.inference_mode()
def generate_stream_traininternlm(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    from fastchat.serve.inference import prepare_logits_processor

    if hasattr(model, "device"):
        device = model.device

    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    logprobs = params.get("logprobs", None)  # FIXME: Support logprobs>1.
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    # if tokenizer.eos_token_id not in stop_token_ids:
    #     stop_token_ids.append(tokenizer.eos_token_id)
        

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids

    max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    start_ids = torch.as_tensor([input_ids], device=device)
    bos_token_id = tokenizer.bos_token_id
    
    has_bos = torch.all(start_ids[:, 0].eq(bos_token_id))
    
    # attention_mask = torch.ones_like(start_ids).eq(0)
    inference_params = InferenceParams(
            max_sequence_len=context_len,
            max_batch_size=1,
            sequence_len_offset=0,
            batch_size_offset=0,
            key_value_memory_dict=None,
            lengths_per_sample=None,
            attention_mask=None,
        )

    past_key_values = out = None
    token_logprobs = [None]  # The first token has no logprobs.
    sent_interrupt = False
    finish_reason = None
    stopped = False
    for i in range(max_new_tokens):
        attention_ids = torch.as_tensor([output_ids], device=device)
        if has_bos:
            bos_pos = torch.where(attention_ids.eq(bos_token_id), 1, 0)
            bos_sum = bos_pos.cumsum(dim=-1)
            bos_pos = torch.where(bos_sum.eq(bos_sum[:, -1:]), 0, 1)
            to_atten_x = bos_pos[:, :, None]
            to_atten_y = bos_pos[:, None, :]
            # attention_mask = torch.einsum('bno,bom->bnm', to_atten_x, to_atten_y).eq(1)
        else:
            bos_pos = torch.where(attention_ids.eq(bos_token_id), 1, 0)
            to_atten_x = bos_pos[:, :, None]
            to_atten_y = bos_pos[:, None, :]
            # attention_mask = torch.einsum('bno,bom->bnm', to_atten_x, to_atten_y).eq(1)
        inference_params.attention_mask = torch.logical_or(to_atten_x, to_atten_y).eq(1)
        if i == 0:  # prefill
            
            out = model(input_ids=start_ids, inference_params=inference_params)
            logits = out
            inference_params.sequence_len_offset += start_ids.size(1)
            # past_key_values = out.past_key_values

            if logprobs is not None:
                # Prefull logprobs for the prompt.
                shift_input_ids = start_ids[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                for label_id, logit in zip(
                    shift_input_ids[0].tolist(), shift_logits[0]
                ):
                    token_logprobs.append(logit[label_id])
        else:  # decoding
            out = model(
                input_ids=torch.as_tensor(
                    [[token] if not sent_interrupt else output_ids],
                    device=device,
                ),
                inference_params=inference_params
                # past_key_values=past_key_values if not sent_interrupt else None,
            )
            inference_params.sequence_len_offset += 1
            sent_interrupt = False
            logits = out
            # past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)
        if logprobs is not None:
            # Cannot use last_token_logits because logprobs is based on raw logits.
            token_logprobs.append(
                torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
            )

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            ret_logprobs = None
            if logprobs is not None:
                ret_logprobs = {
                    "text_offset": [],
                    "tokens": [
                        tokenizer.decode(token)
                        for token in (
                            output_ids if echo else output_ids[input_echo_len:]
                        )
                    ],
                    "token_logprobs": token_logprobs
                    if echo
                    else token_logprobs[input_echo_len:],
                    "top_logprobs": [{}]
                    * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                }
                # Compute text_offset
                curr_pos = 0
                for text in ret_logprobs["tokens"]:
                    ret_logprobs["text_offset"].append(curr_pos)
                    curr_pos += len(text)

            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "logprobs": ret_logprobs,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    else:
        finish_reason = "length"

    if stopped:
        finish_reason = "stop"

    yield {
        "text": output,
        "logprobs": ret_logprobs,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()