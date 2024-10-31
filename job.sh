#!/bin/sh

# Step 1. Generate Distributional Survey Results on OQA, NYT, GlobalValues

## OQA  
# a. Verbalize
python lm_steering.py --task 0 1 3 \
     --dataset opinionqa \
     --demographic_groups  NONE \
     --models llama3-70b gpt-3.5-turbo-0125 anthropic_haiku anthropic_opus gpt-4 \
     --output_type express_distribution; 
# b. Sequence
python lm_steering.py --task 0 1 3 \
     --dataset opinionqa \
     --demographic_groups  POLPARTY SEX \
     --models llama3-70b gpt-3.5-turbo-0125 anthropic_haiku anthropic_opus gpt-4 \
     --output_type sequence; 
# c. Model logprobs
python lm_steering.py --task 0 1 3 \
     --dataset opinionqa \
     --demographic_groups  POLPARTY SEX \
     --models llama3-70b gpt-3.5-turbo-0125 anthropic_haiku anthropic_opus gpt-4 \
     --output_type model_logprobs; 



## NYT
# a. Verbalize
python lm_steering.py --task 0 1 3 \
     --dataset nytimes \
     --demographic_groups  NONE \
     --models llama3-70b gpt-3.5-turbo-0125 anthropic_haiku anthropic_opus gpt-4 \
     --output_type express_distribution; 
# b. Sequence
python lm_steering.py --task 0 1 3 \
     --dataset nytimes \
     --demographic_groups  POLPARTY SEX \
     --models llama3-70b gpt-3.5-turbo-0125 anthropic_haiku anthropic_opus gpt-4 \
     --output_type sequence; 
# c. Model logprobs
python lm_steering.py --task 0 1 3 \
     --dataset nytimes \
     --demographic_groups  POLPARTY SEX \
     --models llama3-70b gpt-3.5-turbo-0125 anthropic_haiku anthropic_opus gpt-4 \
     --output_type model_logprobs; 


## GlobalValues
# a. Verbalize
python lm_steering.py --task 1 3 \
     --demographic_groups globalvalues \
     --models gpt-3.5-turbo-0125 gpt-4 anthropic_opus anthropic_haiku llama3-70b \
     --output_type express_distribution \
     --dataset global_values \
     --task3_type easy_hard; 
# b. Sequence
python lm_steering.py --task 1 3 \
     --demographic_groups globalvalues \
     --models gpt-3.5-turbo-0125 gpt-4 anthropic_opus anthropic_haiku llama3-70b    \
     --output_type sequence \
     --dataset global_values \
     --task3_type easy_hard; 
# c. Model logprobs
python lm_steering.py --task 1 3 \
     --demographic_groups globalvalues \
     --models gpt-3.5-turbo-0125 gpt-4 anthropic_opus anthropic_haiku llama3-70b \
     --output_type model_logprobs \
     --dataset global_values \
     --task3_type easy_hard; 
     
# Step 2. Rescale model log_probs 

python temperature_scaling.py

# Step 3. Evaluate Distributional Survey Results on OQA, NYT, GlobalValues

python lm_steering_eval.py
