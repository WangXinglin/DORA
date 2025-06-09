
#  "Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct"  "Qwen2.5-1.5B-Instruct" 
#!/bin/bash

prm_model="Qwen2.5-Math-PRM-7B"
policy_model="Llama-3.2-1B-Instruct"

for dataset_name in "MATH500"; do
    for approach in "dora"; do
        for seed in 0 1 2 3 4; do
            for n in 16 32 64 128 256; do
                python /mnt/public/usr/yourpath/search-and-learn/scripts/merge_jsonl.py  \
                --n=$n \
                --seed=$seed \
                --balance_alpha="0.01" \
                --prm_model=$prm_model \
                --policy_model=$policy_model \
                --approach=$approach \
                --dataset_name=$dataset_name
            done
        done
    done
done
