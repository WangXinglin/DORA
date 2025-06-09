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
import copy
import logging
from dataclasses import dataclass

import numpy as np
from vllm import LLM, SamplingParams

logger = logging.getLogger()


def build_conv(
    prompt: str, response: str | None, system_prompt: str
) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if response != "":
        conversation.append({"role": "assistant", "content": response})

    return conversation


def last(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return x[-1]


def list_mean(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return np.mean(x)


@dataclass
class Beam:
    prompt: str
    index: int
    current_text: str | None
    next_texts: list[str] | None
    lookahead_texts: list[str] | None
    stop_reasons: list[str | None] | None
    best_scores: list[float]  # the PRM scores
    all_scores: list[list[float]]  # all PRM scores
    store_tokens: list[int] | None
    previous_text: str | None
    pruned: False
    history: list[str]
    history_embedding: list[list[float]]
    completed: bool = False
    completion_tokens: int = 0
    previous_cluster: int = -1


@dataclass
class GenResult:
    index: int
    initial_prompt: str
    first_step_text: str
    first_step_stop_reason: str
    lookahead_text: str
    stop_reason: str | None
    completion_tokens: int = 0


def generate_k_steps(
    templated_convs,
    lookahead_steps: int,
    llm: LLM,
    sampling_params: SamplingParams,
    beam_width: int,
    max_tokens_per_step,
    left_completion_tokens = []
) -> list[Beam]:
    gen_results = []
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text,
                first_step_text="",
                lookahead_text="",
                stop_reason=None,
                completion_tokens=0,
                first_step_stop_reason=None,
            )
            gen_results.append(gen_result)

    gen_sampling_params = copy.deepcopy(sampling_params)
    gen_sampling_params.max_tokens = max_tokens_per_step # incase of repeated generation
    gen_sampling_params_list = []
    for i in range(len(left_completion_tokens)):
        for j in range(beam_width):
            cur_gen_sampling_params = copy.deepcopy(gen_sampling_params)
            cur_gen_sampling_params.max_tokens = min(left_completion_tokens[i], gen_sampling_params.max_tokens)
            gen_sampling_params_list.append(copy.deepcopy(cur_gen_sampling_params))

    for i in range(lookahead_steps + 1):
        if i == 1:
            gen_sampling_params.temperature = 0.0  # greedy for the rest of the steps
        # get all generations that did not finish with eos
        current_gen = [
            gen_results[j]
            for j in range(len(gen_results))
            if gen_results[j].stop_reason != "EOS"
        ]
        gen_sampling_params_list = [gen_sampling_params_list[j] for j in range(len(gen_results)) if gen_results[j].stop_reason != "EOS"]
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ]

        llm_outputs = llm.generate(gen_prompts, gen_sampling_params_list, use_tqdm=False)
        for gen_result, output in zip(current_gen, llm_outputs):
            gen_text = output.outputs[0].text
            gen_tokens = len(output.outputs[0].token_ids)
            if i == 0:
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].stop_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

            gen_result.lookahead_text = gen_result.lookahead_text + gen_text
            gen_result.stop_reason = output.outputs[0].stop_reason
            gen_result.completion_tokens = gen_tokens
            if gen_result.stop_reason is None:
                gen_result.stop_reason = "EOS"

    outputs: list[Beam] = []

    counter = 0
    for i, text in enumerate(templated_convs):
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        completion_tokens = 0
        store_tokens = []
        
        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            completion_tokens += gen_result.completion_tokens   # for beam search
            store_tokens.append(gen_result.completion_tokens)   # for dvts
            counter += 1

        beam_result = Beam(
            prompt=text,
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
            history_embedding=[],
            completion_tokens=completion_tokens,
            store_tokens=store_tokens
        )
        outputs.append(beam_result)

    return outputs
