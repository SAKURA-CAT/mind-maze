"""
@DATE: 2025-05-27 16:01:37
@File: model.py
@IDE: vscode
@Description:
    自定义Qwen3ForCausalLM，为 generate 方法添加 thinking 模式的文案混淆
"""

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers import (
    PretrainedConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
import os
import torch
from torch import nn
from typing import Optional, Union, TYPE_CHECKING
from transformers.generation.utils import (
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
    GenerateNonBeamOutput,
)
from method import methods

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer


class MMQwen3Config(PretrainedConfig):
    model_type = "mm_qwen3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MMQwen3ForCausalLM(Qwen3ForCausalLM):
    """
    自定义 Qwen3 类型，添加 thinking 模式的文案混淆功能
    暂时只支持 batch_size=1 的情况
    """

    config_class = MMQwen3Config
    THINK_START_ID = 151667
    THINK_END_ID = 151668

    def __init__(self, config: MMQwen3Config):
        super().__init__(config)
        self.mm = config.mm
        self.obfuscate = False

    def generate(self, **kwargs):
        generated_ids = super().generate(**kwargs)
        output_ids = generated_ids[0]
        input_ids = output_ids.unsqueeze(0)
        # -------------------------------- Confusion begins ----------------------------------
        # 1. get think start and end indices
        think_start_index = int(
            (input_ids == self.THINK_START_ID).nonzero(as_tuple=True)[1]
        )
        think_end_index = int(
            (input_ids == self.THINK_END_ID).nonzero(as_tuple=True)[1]
        )
        # 2. if think start and end indices are found, confuse the content between them
        if think_start_index != -1 and think_end_index != -1:
            # 3. get the content between think start and end indices
            think_content = input_ids[:, think_start_index + 1 : think_end_index]
            # 4. confuse the content by reversing it
            confused_content = methods["adaptive_rearrange"](think_content, self.mm)
            # 5. replace the original content with the confused content
            input_ids[:, think_start_index + 1 : think_end_index] = confused_content
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = torch.ones_like(input_ids)
        # ---------------------------------- Confusion ends ----------------------------------
        result = super().generate(**kwargs)
        self.obfuscate = False
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        """
        This part of the code modifies GenerationMixin._sample by adding semantic obfuscation to the "think" section.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs = self._get_initial_cache_position(
            cur_len, input_ids.device, model_kwargs
        )

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(
            model_kwargs, generation_config
        )
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(
                input_ids, generation_config, **model_kwargs
            )
            is_prefill = False
        else:
            is_prefill = True
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )

            # ---------------------------------- Confusion break ----------------------------------
            current_token = int(
                input_ids[0, -1] if batch_size == 1 else input_ids[:, -1]
            )
            if current_token == self.THINK_END_ID and not self.obfuscate:
                self.obfuscate = True
                break
            # ------------------------------------------------------------------------------------

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


# 动态注册
AutoConfig.register("mm_qwen3", MMQwen3Config)
AutoTokenizer.register(MMQwen3Config, AutoTokenizer)
AutoModelForCausalLM.register(MMQwen3Config, MMQwen3ForCausalLM)

MODE_NAME = "./weights/MMQwen3-4B"

if __name__ == "__main__":

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(MODE_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODE_NAME, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    # prepare the model input
    # prompt = (
    #     "Strawberry is a fruit. How many r's are in the English word for strawberry?"
    # )
    prompt = "Hello"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # the result will begin with thinking content in <think></think> tags, followed by the actual response
    print(tokenizer.decode(output_ids, skip_special_tokens=True))
