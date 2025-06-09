import os
import json
from copy import deepcopy
import logging

import torch
# from vllm import LLM

from sal.config import Config
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from copy import deepcopy
from datasets import Dataset
import time
import os
import random
import argparse

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def merge_result_files(result_dir, final_output_file, n, temp, seed, threshold, alpha, embedding, alpha_strategy, approach):
    all_results = []
    seen_items = set()  
    valid_path = []
    
    for filename in os.listdir(result_dir):
        if embedding:
            if "embedding" not in filename:
                continue
        else:
            if "embedding" in filename:
                continue
        if alpha_strategy is None:
            if "strategy" in filename:
                continue
        else:
            if "strategy_{}".format(alpha_strategy) not in filename:
                continue

        if approach == "dora":
            if "alpha_{}".format(alpha) not in filename:
                continue

        if filename.endswith(".jsonl") and f"n_{n}" in filename and f"seed_{seed}" in filename and f"temp_{temp}" in filename and "ori" not in filename and "final" not in filename:
            valid_path.append(filename)
            file_path = os.path.join(result_dir, filename)
            results = []
            with open(file_path, "r") as f:
                for line in f:
                    result = json.loads(line.strip())
                    results.append(result)

                for entry in results:
                    item_idx = entry.get("idx")
                    if item_idx not in seen_items:
                        all_results.append(deepcopy(entry))
                        seen_items.add(item_idx)

    return all_results


def main():
    main_parser = argparse.ArgumentParser()

    main_parser.add_argument('--n', type=int, default=16)
    main_parser.add_argument('--seed', type=int, default=0)
    main_parser.add_argument('--prm_model', type=str, default="Qwen2.5-Math-PRM-7B")
    main_parser.add_argument('--policy_model', type=str, default="Llama-3.2-1B-Instruct")
    main_parser.add_argument('--approach', type=str, default="best_of_n")
    main_parser.add_argument('--embedding_threshold', type=str, default="0.95")
    main_parser.add_argument('--balance_alpha', type=str, default="0.01")
    main_parser.add_argument('--alpha_strategy', type=str, default=None)
    main_parser.add_argument('--dataset_name', type=str, default="MATH500")

    args = main_parser.parse_args()

    n = args.n
    temp = 0.8
    prm_model = args.prm_model
    policy_model = args.policy_model
    seed = args.seed
    approach = args.approach
    embedding_threshold = args.embedding_threshold
    alpha = args.balance_alpha
    alpha_strategy = args.alpha_strategy
    dataset_name = args.dataset_name

    if approach == "dora":
        embedding = True
    else:
        embedding = False
    
    # used to merge all parallel running results
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    result_dir = "/mnt/public/usr/yourpath/search-and-learn/result/{}/{}/{}/{}".format(dataset_name, prm_model, policy_model, approach)
    
    if approach == "dora":
        if alpha_strategy is None:
            final_output_file = "/mnt/public/usr/yourpath/search-and-learn/result/{}/{}/{}/{}/final_embedding_results_n_{}_alpha_{}_seed_{}_temp_{}.jsonl".format(dataset_name, prm_model, policy_model, approach, n, alpha, seed, temp)
        else:
            final_output_file = "/mnt/public/usr/yourpath/search-and-learn/result/{}/{}/{}/{}/final_embedding_results_n_{}_alpha_{}_strategy_{}_seed_{}_temp_{}.jsonl".format(dataset_name, prm_model, policy_model, approach, n, alpha, alpha_strategy, seed, temp)
    else:
        final_output_file = "/mnt/public/usr/yourpath/search-and-learn/result/{}/{}/{}/{}/final_results_n_{}_seed_{}_temp_{}.jsonl".format(dataset_name, prm_model, policy_model, approach, n, seed, temp)

        
    merged_result = merge_result_files(result_dir, final_output_file, n, temp, seed, embedding_threshold, alpha, embedding, alpha_strategy, approach)

    config.n = n
    config.num_proc = 12
    config.approach = approach
    config.output_file = final_output_file

    dataset = Dataset.from_list(merged_result)
    dataset = score(dataset, config)

    save_dataset(dataset, config)
    # logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
