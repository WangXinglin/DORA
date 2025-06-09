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
import os

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor


        
def evaluate(benchmark: str, dataset_path: str, dataset_config: str = None, dataset_split: str = "test", dataset_col: str = "pred", samples: list=None, max_num_samples=None, show_level=False):
    samples = load_dataset('json', data_files=dataset_path, split=dataset_split) #load_dataset(dataset_path, name=dataset_config, split=)
    
    if "idx" not in samples.column_names:
        samples = samples.map(lambda x, idx: {"idx": idx}, with_indices=True)
        
    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]

    def parse_gt(x):
        x['gt_cot'], x['gt'] = parse_ground_truth(x, benchmark)
        return x

    samples = samples.map(parse_gt, desc="Parsing ground truth", num_proc=12, load_from_cache_file=False)
    samples = samples.map(extract_answer_map, fn_kwargs={"data_name": benchmark, "col": dataset_col}, desc="Parsing predictions", num_proc=12, load_from_cache_file=False)
    # if show_level:
    #     params = [(idx, pred, gt, level) for idx, pred, gt in zip(samples['idx'], samples['pred'], samples['gt'], samples["level"])]
    # else:
    params = [(idx, pred, gt) for idx, pred, gt in zip(samples['idx'], samples['pred'], samples['gt'])]

    scores = []
    timeout_cnt = 0
    idx = 0
    level_scores = {"1":[], "2": [], "3": [], "4": [], "5":[]}

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                    if show_level:
                        level_scores[str(samples[idx]["level"])].append(result)
                        idx+=1
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

    mean_score = np.mean(scores) * 100

    if show_level:
        mean_level_scores = {}
        for key in level_scores.keys():
            mean_level_scores[key] = np.mean(level_scores[key])*100

        result_json = {
            "num_samples": len(samples),
            "num_scores": len(scores),
            "timeout_samples": timeout_cnt,
            "acc": mean_score,
            "acc_level": mean_level_scores
        }
    else:
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
    parser.add_argument("--benchmark", type=str, default="math")
    parser.add_argument("--dataset_dir", type=str, default="/mnt/public/usr/wangxinglin/search-and-learn/result/MATH500/Qwen2.5-Math-PRM-7B/Llama-3.2-1B-Instruct/beam_search")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--show_level", type=bool, default=True)
    parser.add_argument("--voting_n", type=int, nargs='+', default=[1, 8, 16, 32, 64, 128, 256]) #8, 16, 32, 64, 128, 256
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    for filename in os.listdir(args.dataset_dir):
        if "final_results" in filename:
            cur_file_path = os.path.join(args.dataset_dir, filename)
            cur_n = int(filename.split("n_")[-1].split("_")[0])
            args.voting_n = [cur_n]
            if args.show_level:
                data = {"n": [], "acc_naive": [], "acc_naive_level": [], "acc_weighted": [], "acc_weighted_level": [], "acc_maj": [], "acc_maj_level": []}
            else:
                data = {"n": [], "acc_naive": [], "acc_weighted": [], "acc_maj": []}
            result_path = cur_file_path.split(".jsonl")[0]+"_result.json"

            def evaluate_for_n(n):
                local_data = {"n": n, "acc_naive": None, "acc_weighted": None, "acc_maj": None}
                for agg in ["naive", "weighted", "maj"]:
                    _, scores = evaluate(
                        benchmark=args.benchmark,
                        dataset_path=cur_file_path,
                        dataset_config=args.dataset_config,
                        dataset_split=args.dataset_split,
                        dataset_col=f"pred_{agg}@{n}",
                        max_num_samples=args.max_num_samples,
                        show_level=args.show_level,
                    )
                    local_data[f"acc_{agg}"] = scores["acc"]
                    if args.show_level:
                        local_data[f"acc_{agg}_level"] = scores["acc_level"]
                return local_data

            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(evaluate_for_n, n): n for n in args.voting_n}
                with tqdm(total=len(futures), desc="Evaluating voting_n") as progress_bar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            data["n"].append(result["n"])
                            data["acc_naive"].append(result["acc_naive"])
                            data["acc_weighted"].append(result["acc_weighted"])
                            data["acc_maj"].append(result["acc_maj"])
                            if args.show_level:
                                data["acc_naive_level"].append(result["acc_naive_level"])
                                data["acc_weighted_level"].append(result["acc_weighted_level"])
                                data["acc_maj_level"].append(result["acc_maj_level"])
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
