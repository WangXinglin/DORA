import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd
from datasets import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor
from copy import deepcopy

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def evaluate(benchmark: str, dataset_path: str, dataset_config: str = None, dataset_split: str = "test", dataset_col: str = "pred", samples: list=None, max_num_samples=None, show_level=False):
    #samples = load_dataset('json', data_files=dataset_path, split=dataset_split) #load_dataset(dataset_path, name=dataset_config, split=)
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as file:
        for line in file:
            sub_data = json.loads(line)
            samples.append(deepcopy(sub_data))
        file.close()

    samples = Dataset.from_list(samples)
    if "idx" not in samples.column_names:
        samples = samples.map(lambda x, idx: {"idx": idx}, with_indices=True)
        
    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]

    def parse_gt(x):
        x['gt_cot'], x['gt'] = parse_ground_truth(x, benchmark)
        return x

    samples = samples.map(parse_gt, desc="Parsing ground truth", num_proc=12, load_from_cache_file=False)
    samples = samples.map(extract_answer_multiple_map, fn_kwargs={"data_name": benchmark, "col": dataset_col}, desc="Parsing predictions", num_proc=12, load_from_cache_file=False)
    #samples = samples.map(extract_answer_map, fn_kwargs={"data_name": benchmark, "col": dataset_col}, desc="Parsing predictions", num_proc=12, load_from_cache_file=False)
    # if show_level:
    #     params = [(idx, pred, gt, level) for idx, pred, gt in zip(samples['idx'], samples['pred'], samples['gt'], samples["level"])]
    # else:
    # params = [(idx, pred, gt) for idx, pred, gt in zip(samples['idx'], samples['pred'], samples['gt'])]
    # params = [(i, pred, gt) for i, (pred, gt) in enumerate(zip(samples['pred'], samples['gt']))]
    params = []
    for i in range(len(samples['gt'])):
        for j in range(len(samples[f'pred'][i])):
            params.append((i, samples[f'pred'][i][j], samples['gt'][i]))

    scores = []
    timeout_cnt = 0
    all_idx = []
    with ProcessPool(max_workers=12) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(params), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result, idx = next(iterator)
                    all_idx.append(idx)
                    scores.append(result)
                
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    # print(all_idx)
    mean_score = np.mean(scores) * 100


    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "acc": mean_score
    }
    print(result_json)
    return samples, result_json



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prm_model', type=str, default="Qwen2.5-Math-PRM-7B")
    parser.add_argument('--policy_model', type=str, default="Llama-3.2-3B-Instruct")
    parser.add_argument('--approach', type=str, default="rebase_first")
    parser.add_argument('--embedding_threshold', type=str, default="0.95")
    parser.add_argument('--alpha', type=str, default="0.01")
    parser.add_argument('--alpha_strategy', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument("--benchmark", type=str, default="aime")
    parser.add_argument("--dataset_name", type=str, default="AIME2024")
    #parser.add_argument("--dataset_path", type=str, default="/mnt/public/usr/wangxinglin/search-and-learn/result/MATH500/Qwen2.5-Math-PRM-7B/Llama-3.2-1B-Instruct/diverse_beam_search/final_embedding_results_n_64_alpha_0_seed_0_temp_0.8.jsonl")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--show_level", type=str2bool, default=True)
    parser.add_argument("--voting_n", type=int, nargs='+', default=[64]) #8, 16, 32, 64, 128, 256
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.alpha_strategy is None:
        if args.approach == "quality_first_search":
            args.dataset_path = "/mnt/public/usr/wangxinglin/search-and-learn/result/{}/{}/{}/{}/final_results_n_{}_threshold_{}_seed_{}_temp_0.8.jsonl".format(args.dataset_name, args.prm_model, args.policy_model, args.approach, args.n, args.embedding_threshold, args.seed)
        elif args.approach == "diverse_beam_search" or args.approach == "balance_beam_search" or args.approach == "rebase_diverse" or args.approach == "rebase_first":
            args.dataset_path = "/mnt/public/usr/wangxinglin/search-and-learn/result/{}/{}/{}/{}/final_embedding_results_n_{}_alpha_{}_seed_{}_temp_0.8.jsonl".format(args.dataset_name, args.prm_model, args.policy_model, args.approach, args.n, args.alpha, args.seed)
        else:
            args.dataset_path = "/mnt/public/usr/wangxinglin/search-and-learn/result/{}/{}/{}/{}/final_results_n_{}_seed_{}_temp_0.8.jsonl".format(args.dataset_name, args.prm_model, args.policy_model, args.approach, args.n, args.seed)
    else:
        args.dataset_path = "/mnt/public/usr/wangxinglin/search-and-learn/result/{}/{}/{}/{}/final_embedding_results_n_{}_alpha_{}_strategy_{}_seed_{}_temp_0.8.jsonl".format(args.dataset_name, args.prm_model, args.policy_model, args.approach, args.n, args.alpha, args.alpha_strategy, args.seed)

    
    data = {"n": [], "pass_rate": []}
    result_path = args.dataset_path.split(".jsonl")[0]+"_result_pass_rate.json"

    def evaluate_for_n(n):
        local_data = {"n": n, "pass_rate": 0}
        
        _, scores = evaluate(
            benchmark=args.benchmark,
            dataset_path=args.dataset_path,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            dataset_col=f"preds@{n}",
            max_num_samples=args.max_num_samples,
            show_level=args.show_level,
        )
        local_data[f"pass_rate"] = scores["acc"]
            
        return local_data

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(evaluate_for_n, n): n for n in args.voting_n}
        with tqdm(total=len(futures), desc="Evaluating voting_n") as progress_bar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    data["n"].append(result["n"])
                    data["pass_rate"].append(result["pass_rate"])
                except Exception as e:
                    print(f"Error processing n={futures[future]}: {e}")
                progress_bar.update(1)

    # Save results
    # ds = Dataset.from_dict(data)
    n_list = data["n"]
    sorted_indices = sorted(range(len(n_list)), key=lambda i: n_list[i])
    sorted_data = {}

    for key, value in data.items():
        sorted_data[key] = [value[i] for i in sorted_indices]

    with open(result_path, "w") as f:
        json.dump(sorted_data, f, indent=4)
        f.close()
    print(f"Results pushed to {result_path}")
    # url = ds.push_to_hub(args.dataset_id, config_name=f"{args.dataset_config}--evals")
    # print(f"Results pushed to {url}")
