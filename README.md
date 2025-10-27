Code and data for NeurIPS2025 paper "Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling"


## Abstract

Test-Time Scaling (TTS) improves the performance of Large Language Models (LLMs) by using additional inference-time computation to explore multiple reasoning paths through search. Yet how to allocate a fixed rollout budget most effectively during search remains underexplored, often resulting in inefficient use of compute at test time. To bridge this gap, we formulate test-time search as a resource allocation problem and derive the optimal allocation strategy that maximizes the probability of obtaining a correct solution under a fixed rollout budget. Within this formulation, we reveal a core limitation of existing search methods: solution-level allocation tends to favor reasoning directions with more candidates, leading to theoretically suboptimal and inefficient use of compute. To address this, we propose Direction-Oriented Resource Allocation (DORA), a provably optimal method that mitigates this bias by decoupling direction quality from candidate count and allocating resources at the direction level. To demonstrate DORA’s effectiveness, we conduct extensive experiments on challenging mathematical reasoning benchmarks including MATH500, AIME2024, and AIME2025. The empirical results show that DORA consistently outperforms strong baselines with comparable computational cost, achieving state-of-the-art accuracy. We hope our findings contribute to a broader understanding of optimal TTS for LLMs.


![1](https://github.com/user-attachments/assets/42e68b1a-2114-43e5-8358-420867337dca)
![2](https://github.com/user-attachments/assets/23761596-097d-4a7a-b351-7e1eb8881e9d)

## Experiments

![3](https://github.com/user-attachments/assets/2c371181-8976-4427-9171-96d582467f62)
![4](https://github.com/user-attachments/assets/bcc9f53b-a8ec-4a52-97c2-a8a73082f16c)
![5](https://github.com/user-attachments/assets/c0a6d871-dbb9-4c46-a87d-50ce063f2679)
![6](https://github.com/user-attachments/assets/6c25e7b9-6ec2-4593-b975-3e2f9a71d090)
