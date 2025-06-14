#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import accumulate

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import torch.nn.functional as F
from sal.config import Config
from modeling_qwen2_rm import Qwen2ForProcessRewardModel
from model_utils.prm_model import PRM_MODEL
from model_utils.io_utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards
from copy import deepcopy


CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902


def batched_math_shepherd_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: list[str],
    batch_size: int,
) -> list[list[float]]:
    output_scores = []
    for i in range(0, len(inputs), batch_size):
        inputs_batch = inputs[i : i + batch_size]
        inputs_batch = tokenizer(inputs_batch, padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            logits = model(**inputs_batch).logits[:, :, CANDIDATE_TOKENS]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores_flat = scores[inputs_batch.input_ids == STEP_TAG_ID].tolist()
            # Split scores into sublist based on number of \n in the input
            step_scores = []
            counter = 0
            for i in range(len(inputs_batch.input_ids)):
                count = inputs_batch.input_ids[i].tolist().count(STEP_TAG_ID)
                step_scores.append(step_scores_flat[counter : counter + count])
                counter += count

        # Store the step scores for this batch
        output_scores.extend(step_scores)

        # Clear GPU memory
        del inputs_batch, logits, scores
        torch.cuda.empty_cache()

    return output_scores


class PRM:
    def __init__(self, search_config: Config, **model_kwargs):
        self.search_config = search_config
        self.model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)

    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        raise NotImplementedError

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        raise NotImplementedError


class MathShepherd(PRM):
    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_id = "peiyi9979/math-shepherd-mistral-7b-prm"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # For batched inference
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).eval()
        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        inputs_for_prm = []
        lengths = []
        for question, output in zip(questions, outputs):
            prompt = self.search_config.system_prompt + "\n" + question + "\n"
            special_outputs = [o.replace("\n\n", " ки\n\n") for o in output]
            special_outputs = [
                o + " ки" if o[-2:] != "\n\n" else o for o in special_outputs
            ]
            inputs_for_prm.extend([f"{prompt} {o}" for o in special_outputs])
            lengths.append(len(output))

        # TODO: tokenize each batch independently so there is less padding and faster inference
        output_scores = batched_math_shepherd_inference(
            self.model,
            self.tokenizer,
            inputs_for_prm,
            self.search_config.prm_batch_size,
        )
        cumulative_lengths = list(accumulate(lengths))
        # reshape the output scores to match the input
        output_scores = [
            output_scores[i:j]
            for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
        ]

        # stripped_output_scores = [] TODO: strip out the reward for previous steps
        for output_score, output in zip(output_scores, outputs):
            assert len(output_score) == len(
                output
            ), f"{len(output_score)} != {len(output)}"

        return output_scores


class RLHFFlow(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.search_config.prm_path
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.search_config.prm_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
    ) -> list[list[float]]:
        # if self.search_config.prm_batch_size > 1:   # update here, do not use score_single, it is really slow
        return self._score_batched(questions, outputs, batch_size=self.search_config.prm_batch_size)
        # else: 
        #     return self._score_single(questions, outputs)

    def _score_single(self, questions: list[str], outputs: list[list[str]]):  # single step is quite slow
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                conversation = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        # TODO: add the system prompt like we did for math shepard?
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    if text == "":  # fixed  here
                        continue
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})
                    input_ids = self.tokenizer.apply_chat_template(
                        conversation, return_tensors="pt"
                    ).to(self.model.device)
                    with torch.no_grad():
                        logits = self.model(input_ids).logits[
                            :, -3, self.candidate_tokens
                        ]  # simple version, the +/- is predicted by the '-3' position
                        step_scores = logits.softmax(dim=-1)[
                            :, 0
                        ]  # 0 means the prob of + (1 mean -)
                        # print(scores)
                        single_step_score.append(
                            step_scores[0]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            .item()
                        )

                all_step_scores.append(single_step_score)
            all_scores.append(all_step_scores)
        return all_scores

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.

        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                conversation2 = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    if text == "":  # fixed  here
                        continue
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})

                    # we track to location of the special token with ки in order to extract the scores
                    conversation2.append({"content": text, "role": "user"})
                    conversation2.append({"content": "ки", "role": "assistant"})

                conversations.append(conversation)
                conversations2.append(conversation2)

        output_scores = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)
                
                for i in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores_flat = scores[i, :-1][
                        inputs2_batch[i, 1:] == special_tok_id
                    ].tolist()
                    output_scores.append(step_scores_flat)

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores


class QwenFlow(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.search_config.prm_path, 
        )
        # config = AutoConfig.from_pretrained(self.search_config.prm_path)
        model = Qwen2ForProcessRewardModel.from_pretrained(
            self.search_config.prm_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # config=config,
            **model_kwargs,
        ).eval()

        return model, tokenizer

    def make_step_rewards(self, logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
    
    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
    ) -> list[list[float]]:
        # update here
        return self._score_batched(questions, outputs, self.search_config.prm_batch_size)
        # else:
        #     return self._score_single(questions, outputs)

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.
        
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                ans_list = ans.split("\n\n")
                ans_list = [item for item in ans_list if item!=""]
                ans_prm = "<extra_0>".join(ans_list) + "<extra_0>"
                
                conversation.append({"content": "Please reason step by step, and put your final answer within \\boxed{}.", "role": "system"})
                conversation.append({"content": question, "role": "user"})
                conversation.append({"content": ans_prm, "role": "assistant"})
                conversations.append(conversation)

        output_scores = []
        step_sep_id = self.tokenizer.encode("<extra_0>")[0]
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i: i+batch_size]
            
            input_ids = self.tokenizer.apply_chat_template(convs_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                prm_outputs = self.model(input_ids=input_ids)
                token_masks = (input_ids == step_sep_id)
                step_reward = self.make_step_rewards(prm_outputs[0], token_masks)
                
                # output_scores.append(step_reward)
                for j in range(len(convs_batch)):
                    output_scores.append(step_reward[j])

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores


class SkyworkFlow(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.search_config.prm_path, 
        )
        # config = AutoConfig.from_pretrained(self.search_config.prm_path)
        model = PRM_MODEL.from_pretrained(self.search_config.prm_path, device_map="auto", torch_dtype=torch.bfloat16).eval()
        return model, tokenizer

    def make_step_rewards(self, logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
    
    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
    ) -> list[list[float]]:
        # update here
        return self._score_batched(questions, outputs, self.search_config.prm_batch_size)

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.
        
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        output_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                cur = prepare_input(question, ans, tokenizer=self.tokenizer, step_token="\n\n")
                conversations.append(deepcopy(cur))
        
        for i in range(0, len(conversations), batch_size):
            input_ids, steps, reward_flags = zip(*conversations[i: i+batch_size])
            input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, self.tokenizer.pad_token_id)
            with torch.no_grad():
                _, _, rewards = self.model(input_ids=input_ids.to(self.model.pretrained_model.device), attention_mask=attention_mask.to(self.model.pretrained_model.device), return_probs=True)
                step_rewards = derive_step_rewards(rewards, reward_flags)
                for j in range(len(step_rewards)):
                    output_scores.append(deepcopy(step_rewards[j]))

        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(deepcopy(output_scores[counter]))
                counter += 1
            reshaped_output_scores.append(deepcopy(scores))

        return reshaped_output_scores
        
        # for question, answers in zip(questions, outputs, strict=True):
        #     for ans in answers:
        #         conversation = []
        #         ans_list = ans.split("\n\n")
        #         ans_list = [item for item in ans_list if item!=""]
        #         ans_prm = "<extra_0>".join(ans_list) + "<extra_0>"
                
        #         # conversation.append({"content": "Please reason step by step, and put your final answer within \\boxed{}.", "role": "system"})
        #         conversation.append({"content": question, "role": "user"})
        #         conversation.append({"content": ans_prm, "role": "assistant"})
        #         conversations.append(conversation)

        # output_scores = []
        # step_sep_id = self.tokenizer.encode("<extra_0>")[0]
        # for i in range(0, len(conversations), batch_size):
        #     convs_batch = conversations[i: i+batch_size]
            
        #     input_ids = self.tokenizer.apply_chat_template(convs_batch, padding=True, return_tensors="pt"
        #     ).to(self.model.device)
        #     with torch.no_grad():
        #         prm_outputs = self.model(input_ids=input_ids)
        #         token_masks = (input_ids == step_sep_id)
        #         step_reward = self.make_step_rewards(prm_outputs[0], token_masks)
                
        #         # output_scores.append(step_reward)
        #         for j in range(len(convs_batch)):
        #             output_scores.append(step_reward[j])

        # # reshape the output scores to match the input
        # reshaped_output_scores = []
        # counter = 0
        # for question, answers in zip(questions, outputs):
        #     scores = []
        #     for answer in answers:
        #         scores.append(output_scores[counter])
        #         counter += 1
        #     reshaped_output_scores.append(scores)

        # return reshaped_output_scores

def load_prm(config: Config) -> PRM:
    if config.prm_path.split("/")[-1] == "math-shepherd-mistral-7b-prm":
        return MathShepherd(config)
    if config.prm_path.split("/")[-1] == "Llama3.1-8B-PRM-Deepseek-Data":
        return RLHFFlow(config)
    if config.prm_path.split("/")[-1] == "Qwen2.5-Math-PRM-7B":
        return QwenFlow(config)
    if config.prm_path.split("/")[-1] == "Skywork-PRM-7B":
        return SkyworkFlow(config)
    raise NotImplementedError(f"PRM {config.prm_path} not implemented")
