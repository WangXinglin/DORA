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

import torch
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts, rebase, dora
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from copy import deepcopy
from datasets import Dataset
import time
from datetime import datetime
import os
import random
from FlagEmbedding import BGEM3FlagModel
import torch 
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "rebase": rebase,
    "dora": dora,
}


def save_random_states():
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    else:
        cuda_states = None
    return (py_state, torch_state, cuda_states)

def load_embedding_model(config):
    if config.embedding_path is None:
        return None
    else:
        model = BGEM3FlagModel(config.embedding_path,  use_fp16=False)
        if hasattr(model, "model"):
            model.model.eval()
        else:
            model.eval()
        
        if hasattr(model, "tokenizer"):
            model.tokenizer.parallelism = False
    return model

def generate_random_indices(dataset):
    current_time = time.time()*10000000
    seed = int(current_time)
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return indices


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()
    print(config.filter_duplicates)

    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=False, # False to enable replicate under parallel setting
        seed=config.seed,   # seed for reproduce
        tensor_parallel_size=num_gpus,
        max_model_len=5000,
    )
    if config.approach == "preliminary":
        prm = None
        embedding_model = None
    else:
        prm = load_prm(config)
        embedding_model = load_embedding_model(config)

    dataset = get_dataset(config)
    all_data = []
    all_result = []
    for idx, sample in enumerate(dataset):
        if config.dataset_name.split("/")[-1] not in ["mbpp", "human_eval"]:
            sample["idx"] = idx
        else:
            sample["idx"] = idx
        all_data.append(deepcopy(sample))
    
    original_states = save_random_states() # save current state
    random_indices = generate_random_indices(all_data)
    random.seed(config.seed)

    if config.approach == "dora":
        if embedding_model is not None:
            if config.alpha_strategy is None:
                unique_filename = f"samples_embedding_bge-m3_n_{config.n}_alpha_{config.balance_alpha}_seed_{config.seed}_temp_{config.temperature}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jsonl"
            else:
                unique_filename = f"samples_embedding_bge-m3_n_{config.n}_alpha_{config.balance_alpha}_strategy_{config.alpha_strategy}_seed_{config.seed}_temp_{config.temperature}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jsonl"
        else:
            unique_filename = f"samples_n_{config.n}_alpha_{config.balance_alpha}_seed_{config.seed}_temp_{config.temperature}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jsonl"
    else:
        unique_filename = f"samples_n_{config.n}_seed_{config.seed}_temp_{config.temperature}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jsonl"
    output_file = os.path.join(config.output_dir, unique_filename)
    config.output_file = output_file

    if config.approach == "dora":
        if embedding_model is not None:
            if config.alpha_strategy is None:
                processed_log_file = os.path.join(config.output_dir, f"processed_embedding_bge-m3_items_n_{config.n}_alpha_{config.balance_alpha}_seed_{config.seed}_temp_{config.temperature}.log")
            else:
                processed_log_file = os.path.join(config.output_dir, f"processed_embedding_bge-m3_items_n_{config.n}_alpha_{config.balance_alpha}_strategy_{config.alpha_strategy}_seed_{config.seed}_temp_{config.temperature}.log")
        else:
            processed_log_file = os.path.join(config.output_dir, f"processed_items_n_{config.n}_alpha_{config.balance_alpha}_seed_{config.seed}_temp_{config.temperature}.log")
    else:
        processed_log_file = os.path.join(config.output_dir, f"processed_items_n_{config.n}_seed_{config.seed}_temp_{config.temperature}.log")
    queries = []  
    ids_to_process = [] 
    all_sample_times = []

    for idx in tqdm(random_indices):
        batch = all_data[idx]
        if os.path.exists(processed_log_file):
            with open(processed_log_file, "r") as f:
                processed_items = set(int(line.strip()) for line in f.readlines() if line.strip().isdigit())
        else:
            processed_items = set()

        if idx in processed_items:
            continue
        
        queries.append(deepcopy(batch))
        ids_to_process.append(idx)
        
        if len(queries) >= config.search_batch_size or idx == random_indices[-1]:
            input_batch = {}
            for key in queries[0].keys():
                input_batch[key] = []
            for i in range(len(queries)):
                query = queries[i]
                for key in query.keys():
                    input_batch[key].append(deepcopy(query[key]))

            torch.set_rng_state(original_states[1])
            if original_states[2] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(original_states[2])
            
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)
            start_time = time.time()
            processed_sample = approach_fn(
                                            input_batch,
                                            config=config, 
                                            llm=llm, 
                                            prm=prm,
                                            embedding_model=embedding_model
                                        )
            end_time = time.time()
            elapsed_time = end_time - start_time
            all_sample_times.append(elapsed_time)

            for i in range(len(queries)):
                cur_result = {}
                for key in processed_sample.keys():
                    cur_result[key] = processed_sample[key][i]
                queries[i] |= cur_result
                all_result.append(deepcopy(queries[i]))
                with open(processed_log_file, "a") as f:
                    f.write(f"{ids_to_process[i]}\n")
                    f.close()

            queries.clear()
            ids_to_process.clear()

            cur_dataset = Dataset.from_list(all_result)
            save_dataset(cur_dataset, config)

    print("avg sample time = {}s".format(sum(all_sample_times)/len(all_sample_times)))
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
