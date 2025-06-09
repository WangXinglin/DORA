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


import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv, generate_k_steps
import torch
from copy import deepcopy

logger = logging.getLogger()


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


def get_cos_matrix(completions, config, embedding_model=None):
    
    all_completions = completions
    
    if embedding_model is None:
        all_embeddings = get_embedding(all_completions)
    else:
        all_embeddings = get_embedding_local(all_completions, embedding_model, config)
    all_cos_sim =  compute_cosine_similarity_numpy(all_embeddings)
    return all_cos_sim


def softmax_with_temperature(matrix, temperature=1.0):
    matrix = matrix.astype(np.float64) 
    scaled = matrix / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return (exp_values / exp_values.sum(axis=1, keepdims=True)).astype(np.float32)


def trace_of_softmax_with_temperature(M, temperature=1.0):
    N = softmax_with_temperature(M, temperature)
    trace = np.trace(N)
    return trace

def extract_submatrix_np(matrix, indices):
    return matrix[np.ix_(indices, indices)]


def get_diversity_quality(cos_matrix, quality_score_list, choose_candidate):
    final_candidates = []
    all_candidates = [i for i in range(len(quality_score_list))]
    
    all_quality = sum(quality_score_list)/len(quality_score_list)
    if len(all_candidates) > 1:
        all_diversity = (trace_of_softmax_with_temperature(extract_submatrix_np(cos_matrix, all_candidates), 0.1) / len(all_candidates) - 1/len(all_candidates)) / (1-1/len(all_candidates))
    else:
        all_diversity = 0
    #all_diversity = (trace_of_softmax_with_temperature(extract_submatrix_np(cos_matrix, all_candidates), 0.1) / len(all_candidates) - 1/len(all_candidates)) / (1-1/len(all_candidates))
    final_quality = sum([quality_score_list[idx] for idx in choose_candidate])/len(choose_candidate)
    if len(choose_candidate) > 1:
        final_diversity = (trace_of_softmax_with_temperature(extract_submatrix_np(cos_matrix, choose_candidate), 0.1) / len(choose_candidate) - 1/len(choose_candidate)) / (1-1/len(choose_candidate))
    else:
        final_diversity = 0
        
    return final_quality, final_diversity, all_quality, all_diversity


def _dvts(batch_of_prompts: list[str], config: Config, llm: LLM, prm: PRM, embedding_model=None):
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=[
            "\n\n"
        ],  # we consider that a step in the problem is indicated by a double newline
        include_stop_str_in_output=True,
        n=1,
    )

    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n_beams):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    best_scores=[0.0],
                    all_scores=[],
                    previous_text=None,
                    pruned=False,
                    stop_reasons=None,
                    history=[],
                    completion_tokens=0,
                    store_tokens=[],
                    history_embedding=[]
                )
            )

    final_quality_list = []
    final_diversity_list = []
    all_quality_list = []
    all_diversity_list = []

    for i in tqdm(range(config.num_iterations), desc="DVTS iterations"):
        # generation
        gen_beams = [b for b in beams if not b.pruned]
        if len(gen_beams) == 0:
            break

        if i == config.num_iterations - 1:
            # last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens, #config.max_tokens, #incase of too long generation for last step
                top_p=config.top_p,
                n=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in gen_beams
        ]
        
        active_left_completion_tokens = [config.max_tokens - b.completion_tokens for b in gen_beams]  ####
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()
        # TODO: set the augmented template from a file
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
                templated_convs, lookahead, llm, sampling_params, config.beam_width, config.max_tokens, active_left_completion_tokens
            )
        else:
            gen_results = generate_k_steps(
                templated_convs, lookahead, llm, sampling_params, config.beam_width, config.max_tokens_per_step, active_left_completion_tokens
            )

        prompts, completions = [], []
        all_completions = []
        for beam, gen_result in zip(gen_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.store_tokens = gen_result.store_tokens
            if len(beam.next_texts) != config.beam_width:
                beam.pruned = True
                # rarely ~1/1000 the model will generate few beams than expected. #TODO: investigate why
                logger.warning(
                    f"beam {beam.index} has {len(beam.next_texts)} completions"
                )

            prompts.append(beam.prompt)
            completions.append([beam.current_text + t for t in beam.lookahead_texts])
            all_completions.extend([beam.current_text + t for t in beam.lookahead_texts])
            

        # scoring and chose best generation per beam TODO: add option for selection across beams within the same prompt

        all_scores = prm.score(prompts, completions)

        all_agg_scores = []
        # cos_matrix = get_cos_matrix(all_completions, config, embedding_model)

        choose_indices = []
        count = 0
        for beam, scores in zip(gen_beams, all_scores, strict=True):
            agg_scores = [aggregate_scores(s, config.agg_strategy) for s in scores]
            all_agg_scores += agg_scores
            for idx in range(len(agg_scores)):
                if i!=config.num_iterations - 1 and beam.stop_reasons[idx] == "EOS" and beam.store_tokens[idx] == config.max_tokens_per_step:  # uncompleted step, drop it
                    agg_scores[idx] = 0

            best_score_ind = np.argmax(agg_scores)
            choose_indices.append(count+best_score_ind)
            count += len(agg_scores)
            beam.all_scores = scores
            beam.previous_text = beam.current_text
            beam.current_text = beam.current_text + beam.next_texts[best_score_ind]
            beam.history.append(beam.next_texts[best_score_ind])
            beam.best_scores = scores[best_score_ind]
            beam.completion_tokens += beam.store_tokens[best_score_ind] # particular for dvts
            if (
                beam.next_texts[best_score_ind] == ""
                or beam.stop_reasons[best_score_ind] == "EOS"
                or beam.stop_reasons[best_score_ind] == "length"
                or beam.completion_tokens >= config.max_tokens
            ):
                # stopped on EOS, prune
                beam.pruned = True

        # final_quality, final_diversity, all_quality, all_diversity = get_diversity_quality(cos_matrix, all_agg_scores, choose_indices)
        final_quality_list.append(0)
        final_diversity_list.append(0)
        all_quality_list.append(0)
        all_diversity_list.append(0)

        # filter / prune
        for beam in gen_beams:
            if "boxed{" in beam.current_text:
                beam.pruned = True

    # we need to copy the results from the last iteration in to beam_width beams as otherwise we would only have n/m results
    output: list[Beam] = []
    for beam in beams:
        for i in range(config.beam_width):
            output.append(
                Beam(
                    prompt=beam.prompt,
                    index=beam.index,
                    current_text=beam.previous_text + beam.next_texts[i],
                    next_texts=None,
                    lookahead_texts=None,
                    stop_reasons=None,
                    best_scores=beam.all_scores[i],
                    all_scores=beam.all_scores,
                    previous_text=beam.current_text,
                    pruned=beam.pruned,
                    history=beam.history,
                    completion_tokens=beam.completion_tokens,
                    store_tokens=beam.store_tokens,
                    history_embedding=[]
                )
            )

    return output, final_quality_list, final_diversity_list, all_quality_list, all_diversity_list


def dvts(examples, config: Config, llm: LLM, prm: PRM, embedding_model=None):
    problems = examples["problem"]
    print(problems)
    beam_results, final_quality_list, final_diversity_list, all_quality_list, all_diversity_list = _dvts(problems, config, llm, prm, embedding_model)

    # group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": [], "quality_results": [], "diversity_results":[], "all_quality_results": [], "all_diversity_results":[]}

    for p in problems:
        beams = grouped_results[p]
        results["completions"].append([b.current_text for b in beams])
        results["pred"].append(
            beams[
                np.argmax(
                    [
                        aggregate_scores(b.best_scores, config.agg_strategy)
                        for b in beams
                    ]
                )
            ].current_text
        )
        results["scores"].append([b.best_scores for b in beams])
        results["completion_tokens"].append([b.completion_tokens for b in beams])
        results["quality_results"] = [final_quality_list]
        results["diversity_results"] = [final_diversity_list]
        results["all_quality_results"] = [all_quality_list]
        results["all_diversity_results"] = [all_diversity_list]

    # TODO: construct and store the tree

    return results
