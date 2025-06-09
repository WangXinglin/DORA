pip install vllm==0.6.5
pip install latex2sympy2
pip install word2number
# pip install openai==0.28.0
pip install -U FlagEmbedding

export CONFIG=/mnt/public/usr/yourpath/search-and-learn/recipes/Llama-3.2-1B-Instruct/dvts_qwen.yaml
export prm_path=/mnt/public/usr/yourpath/allmodels/Qwen2.5-Math-PRM-7B
export model_path=/mnt/public/usr/yourpath/allmodels/Llama-3.2-1B-Instruct
export output_dir=/mnt/public/usr/yourpath/search-and-learn/result/MATH500/Qwen2.5-Math-PRM-7B/Llama-3.2-1B-Instruct/dvts


for n in 16 32 64 128 256; do
    for seed in 0 1 2 3 4; do
        python /mnt/public/usr/yourpath/search-and-learn/scripts/test_time_compute.py $CONFIG \
        --n=$n \
        --seed=$seed \
        --prm_path=$prm_path \
        --model_path=$model_path \
        --output_dir=$output_dir
    done
done