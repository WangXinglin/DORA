prm_model="Qwen2.5-Math-PRM-7B"
policy_model="Llama-3.2-3B-Instruct" #"Qwen2.5-1.5B-Instruct" #"Llama-3.2-1B-Instruct"
dataset_name="MATH500" # MATH500
benchmark="math" # math

for approach in "rebase" "beam_search"; do
    for seed in 0 1 2 3 4; do
        for n in 16 32 64 128 256; do
            python /mnt/public/usr/yourpath/search-and-learn/evaluation/evaluate_hf.py  \
            --n=$n \
            --seed=$seed \
            --prm_model=$prm_model \
            --policy_model=$policy_model \
            --approach=$approach \
            --alpha="0.01" \
            --voting_n=$n \
            --dataset_name=$dataset_name \
            --benchmark=$benchmark
        done
    done
done
