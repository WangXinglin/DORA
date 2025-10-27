Code and data for NeurIPS2025 paper "Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling"

## Environment
```bash
pip install -f requirements.txt
```

## Sampling

```bash
bash scripts/run_{$method}_qwen.sh
```

## Merge Sampling Result

```bash
bash scripts/merge_result.sh
```

## Get Accuracy Result

```bash
bash scripts/eval_result.sh
```

## Develop Your Own Sampling Strategy (optional)

Please add your code on ```scripts/sal/search```

## Acknowledgement

We learned a lot and borrowed some code from the following projects when building GDR.

- [search-and-learn](https://github.com/huggingface/search-and-learn)
- [inference_scaling](https://github.com/thu-wyz/inference_scaling) 



## Abstract

Test-Time Scaling (TTS) improves the performance of Large Language Models (LLMs) by using additional inference-time computation to explore multiple reasoning paths through search. Yet how to allocate a fixed rollout budget most effectively during search remains underexplored, often resulting in inefficient use of compute at test time. To bridge this gap, we formulate test-time search as a resource allocation problem and derive the optimal allocation strategy that maximizes the probability of obtaining a correct solution under a fixed rollout budget. Within this formulation, we reveal a core limitation of existing search methods: solution-level allocation tends to favor reasoning directions with more candidates, leading to theoretically suboptimal and inefficient use of compute. To address this, we propose Direction-Oriented Resource Allocation (DORA), a provably optimal method that mitigates this bias by decoupling direction quality from candidate count and allocating resources at the direction level. To demonstrate DORA’s effectiveness, we conduct extensive experiments on challenging mathematical reasoning benchmarks including MATH500, AIME2024, and AIME2025. The empirical results show that DORA consistently outperforms strong baselines with comparable computational cost, achieving state-of-the-art accuracy. We hope our findings contribute to a broader understanding of optimal TTS for LLMs.



![1](https://github.com/user-attachments/assets/d4fe50cf-fa87-465e-97af-48b2473b3118)


![2](https://github.com/user-attachments/assets/9dd0746f-a04a-475e-b47d-b540587f41a9)



