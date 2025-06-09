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
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM

from .utils import Beam, build_conv, generate_k_steps, last

logger = logging.getLogger()
from sal.utils.score import aggregate_scores

import requests
import os
import json
import time 
import math
from copy import deepcopy
import torch

def get_embedding_local(all_completions, embedding_model, config):
    embeddings = []
    with torch.no_grad():  
        for completion in all_completions:
            embedding = embedding_model.encode(completion, 
                                    batch_size=1, 
                                    max_length=config.max_tokens, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                    )['dense_vecs']
            embeddings.append(deepcopy(embedding))
    
    return embeddings


def compute_cosine_similarity_numpy(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float64)  
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)  
    normalized = embeddings / norms
    cosine_sim = np.dot(normalized, normalized.T)
    return cosine_sim.astype(np.float32)  

def average_embedding(embeddings):
    embeddings_array = np.array(embeddings)
    avg_embedding = np.mean(embeddings_array, axis=0)
    
    return avg_embedding


def get_cos_matrix(all_beams, config, embedding_model=None):
    
    all_completions = []
    for beam in all_beams:
        all_completions.append(beam.current_text)
    if embedding_model is None:
        all_embeddings = get_embedding(all_completions)
    else:
        all_embeddings = get_embedding_local(all_completions, embedding_model, config)
    all_cos_sim =  compute_cosine_similarity_numpy(all_embeddings)
    return all_cos_sim


def softmax_with_temperature_1(matrix, temperature=1.0):
    matrix = matrix.astype(np.float64)
    scaled = matrix / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return (exp_values / exp_values.sum(axis=1, keepdims=True)).astype(np.float32)

def softmax_with_temperature(agg_scores, temperature=0.1):
    scores = np.array([score[0] for score in agg_scores], dtype=np.float64)
    scaled_scores = scores / temperature
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    softmax_scores = exp_scores / np.sum(exp_scores)
    return [float(score) for score in softmax_scores]


def get_diversity_score(M, temperature=1.0):
    N = softmax_with_temperature_1(M, temperature)
    diagonal_elements = [N[i][i] for i in range(len(N))]
    return diagonal_elements


def trace_of_softmax_with_temperature(M, temperature=1.0):
    N = softmax_with_temperature_1(M, temperature)
    trace = np.trace(N)
    return trace

def extract_submatrix_np(matrix, indices):
    return matrix[np.ix_(indices, indices)]


def get_diversity_quality(cos_matrix, quality_score_list, choose_candidate):
    final_candidates = []
    quality_score_list = [item[0] for item in quality_score_list]
    all_candidates = [i for i in range(len(quality_score_list))]
    
    all_quality = sum(quality_score_list)/len(quality_score_list)
    if len(all_candidates) > 1:
        all_diversity = (trace_of_softmax_with_temperature(extract_submatrix_np(cos_matrix, all_candidates), 0.1) / len(all_candidates) - 1/len(all_candidates)) / (1-1/len(all_candidates))
    else:
        all_diversity = 0
    final_quality = sum([quality_score_list[idx] for idx in choose_candidate])/len(choose_candidate)
    if len(choose_candidate) > 1:
        final_diversity = (trace_of_softmax_with_temperature(extract_submatrix_np(cos_matrix, choose_candidate), 0.1) / len(choose_candidate) - 1/len(choose_candidate)) / (1-1/len(choose_candidate))
    else:
        final_diversity = 0

    return final_quality, final_diversity, all_quality, all_diversity

def get_diversity_quality_all(cos_matrix, quality_score_list):
    quality_score_list = [item[0] for item in quality_score_list]
    all_candidates = [i for i in range(len(quality_score_list))]
    all_quality = sum(quality_score_list)/len(quality_score_list)
    if len(all_candidates) > 1:
        all_diversity = (trace_of_softmax_with_temperature(extract_submatrix_np(cos_matrix, all_candidates), 0.1) / len(all_candidates) - 1/len(all_candidates)) / (1-1/len(all_candidates))
    else:
        all_diversity = 0

    return all_quality, all_diversity


def _dora(batch_of_prompts, config: Config, llm: LLM, prm: PRM, embedding_model=None) -> list[Beam]:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,  # New flag to track completion
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                    store_tokens=[],
                    history_embedding=[]
                )
            )

    completed_beams: list[Beam] = []
    final_quality_list = []
    final_diversity_list = []
    all_quality_list = []
    all_diversity_list = []

    for i in tqdm(range(config.num_iterations), desc="Rebase iterations"):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            rebase_beams = []
            for k in range(len(allocate_num)):
                if allocate_num[k] > 0:
                    rebase_beams.extend([copy.deepcopy(active_beams[k]) for _ in range(allocate_num[k])])
            active_beams = rebase_beams #[b for b in active_beams if not b.pruned]

        # Duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            logger.debug(
                f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
            )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != config.n:
                raise ValueError(
                    f"Expected {config.n} active beams, but got {len(active_beams)}"
                )

        if i == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens, # in case of OOM
                top_p=config.top_p,
                n=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]
        active_left_completion_tokens = [config.max_tokens - b.completion_tokens for b in active_beams]  ####
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead
        if i == config.num_iterations - 1:
            gen_results = generate_k_steps(
                templated_convs, lookahead, llm, sampling_params, 1, config.max_tokens, active_left_completion_tokens # 1 as beam_width is reasonable, as the authors process the logic out of the function
            )
        else:
            gen_results = generate_k_steps(
                templated_convs, lookahead, llm, sampling_params, 1, config.max_tokens_per_step, active_left_completion_tokens # 1 as beam_width is reasonable, as the authors process the logic out of the function
            )

        prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])

            if (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
                or beam.completion_tokens >= config.max_tokens
            ):
                if i == config.num_iterations - 1:
                    beam.completed = True
                    completed_beams.append(beam)
                else:
                    if gen_result.completion_tokens == config.max_tokens_per_step:
                        beam.pruned = True
                        beam.completed = True
                    else:
                        beam.completed = True
                        completed_beams.append(beam)
            prompts.append(beam.prompt)
            completions.append([beam.current_text])

        scores = prm.score(prompts, completions)

        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]

        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0]

        # Now filter active_beams and agg_scores for beams that are completed
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            break


        # realize SOFTMAX and duplicate here
        all_agg_scores = deepcopy(agg_scores)
        cos_matrix = get_cos_matrix(active_beams, config, embedding_model)
        all_quality, all_diversity = get_diversity_quality_all(cos_matrix, all_agg_scores)

        softmax_agg_scores = softmax_with_temperature(agg_scores, config.rebase_temperature)

        allocate_num = [int(item) for item in np.round(np.array(softmax_agg_scores) * config.n)]

        active_beams = [active_beams[j] for j in range(len(active_beams)) if allocate_num[j]>0]
        agg_scores = [agg_scores[j] for j in range(len(agg_scores)) if allocate_num[j]>0]
        softmax_agg_scores = softmax_with_temperature(agg_scores, config.rebase_temperature)

        cos_matrix = extract_submatrix_np(all_cos_matrix, choose_indices)
        diversity_score_list = get_diversity_score(cos_matrix, config.balance_alpha)

        balance_score = [diversity_score_list[j]*softmax_agg_scores[j] for j in range(len(softmax_agg_scores))]
        final_score = [item/sum(balance_score) for item in balance_score]
        # get diversity score here
        allocate_num = [int(item) for item in np.round(np.array(final_score) * config.n)]

        choose_indices = [j for j in range(len(allocate_num)) if allocate_num[j]>0]
        final_quality, final_diversity, _, _ = get_diversity_quality(cos_matrix, agg_scores, choose_indices)
        final_quality_list.append(final_quality)
        final_diversity_list.append(final_diversity)
        all_quality_list.append(all_quality)
        all_diversity_list.append(all_diversity)

        for idx, beam in enumerate(active_beams):
            if allocate_num[idx] == 0:
                beam.pruned = True
        
        if len(completed_beams) >= config.n:
            break

    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    return completed_beams, final_quality_list, final_diversity_list, all_quality_list, all_diversity_list


def dora(examples, config: Config, llm: LLM, prm: PRM, embedding_model=None):
    problems = examples["problem"]
    print(problems)
    beam_results, final_quality_list, final_diversity_list, all_quality_list, all_diversity_list = _dora(problems, config, llm, prm, embedding_model)

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": [], "quality_results": [], "diversity_results":[], "all_quality_results": [], "all_diversity_results":[]}

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])
        results["quality_results"] = [final_quality_list]
        results["diversity_results"] = [final_diversity_list]
        results["all_quality_results"] = [all_quality_list]
        results["all_diversity_results"] = [all_diversity_list]

    return results
