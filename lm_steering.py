import os
import openai
import json
import numpy as np
import argparse
from utils import *
from collections import defaultdict

openai.api_key=os.environ.get('OPENAI_API_KEY')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_type', type=str, default='sequence', choices=['sequence', 'model_logprobs', 'express_distribution', 'incontext_avg'])
    parser.add_argument('--wave', type=str, default='Pew_American_Trends_Panel_disagreement_100')
    parser.add_argument('--demographic_groups', nargs='+', default=[])
    parser.add_argument('--dataset', type=str, default='opinionqa',  choices=['opinionqa', 'nytimes', 'globalvalues'])
    parser.add_argument('--models', nargs='+', default=[])
    parser.add_argument('--n_seq', type=int, default=1)
    parser.add_argument('--n_sample', type=int, default=1)
    # parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--task', nargs='+', default=[])
    parser.add_argument('--eval_metric', type=str, default='total_variation')
    parser.add_argument('--task3_type', type=str, default='easy_hard')
    parser.add_argument('--ficticious_group_ablation', action='store_true')
    parser.add_argument('--shuffled_incontext_labels', action='store_true')
    parser.add_argument('--task_align', action='store_true')
    parser.add_argument('--k', type=int, default=1) # number of wave drawing dissimilar question (task 4), number of questions drawn from the wave for the training set (task_align)

    args = parser.parse_args()

    # models = ['gpt-4', 'gpt-3.5-turbo-0125', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']

    if args.dataset=='opinionqa': data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
    else: data_path = '{}/{}'.format(os.getcwd(), args.dataset)

    # for each output type: to do
    # for each model 
    for model in args.models:
        if model=='llama3-8b':
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                # The quantization line
                model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
            )
            
        elif model=='llama3-70b':
            if args.output_type != 'model_logprobs': 
                model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_id,
                    # The quantization line
                    model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
                )
            else: pipeline=None
                
        elif model=='mixtral-8x7b':
            model_id = 'mistralai/Mixtral-8x7B-v0.1'
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                # The quantization line
                model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
            )

        elif model=='phi-2':
            model_id ='microsoft/phi-2'
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                # The quantization line
                model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
            )

        elif model=='llama-2-7b':
            model_id ='meta-llama/Llama-2-7b-chat-hf'
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                # The quantization line
                model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
            )

        elif model=='llama-2-13b':
            model_id ='meta-llama/Llama-2-13b-chat-hf'
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                # The quantization line
                model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
            )

        elif model=='llama-2-70b':
            model_id ='meta-llama/Llama-2-70b-chat-hf'
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                # The quantization line
                model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
            )
        
        elif model=='falcon-1b':
            model_id ='tiiuae/falcon-rw-1b'
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                # The quantization line
                model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
            )
        
        elif model=='falcon-7b':
            model_id ='tiiuae/falcon-7b'
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                # The quantization line
                model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
            )

        elif model=='falcon-40b':
            model_id ='tiiuae/falcon-40b'
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                # The quantization line
                model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
            )
        
        elif model=='falcon-180b':
            model_id ='tiiuae/falcon-180b'
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                # The quantization line
                model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
            )
        
        elif model=='deepseek-coder-1.3b' or model=='deepseek-coder-6.7b' or model=='deepseek-coder-33b':
            pipeline = AutoModelForCausalLM.from_pretrained("deepseek-ai/{}-base".format(model), trust_remote_code=True).cuda()

        else: pipeline=None

        if model=='llama3-70b' and args.output_type =='model_logprobs':

            model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

            # Load the model and tokenizer
            model_llama3 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, load_in_4bit=True)
            tokenizer_llama3 = AutoTokenizer.from_pretrained(model_id)
            pipeline = (model_llama3,tokenizer_llama3)
            
        # for each demographic group
        for wave in args.wave:
                for demographic_group in args.demographic_groups:
                    eval_metric = []
                    for demographic in dem_group_to_dem_mapping[demographic_group]: ### edit this later to grab it from the json
                        for task_num in args.task:
                            print(demographic)
                            print('\n')

                            if args.dataset=='opinionqa': q_IDS, waves = get_q_IDs_waves_disagree(data_path, foldername=args.wave, args=args)
                            elif args.dataset=='globalvalues':
                                q_IDS = get_q_IDs_waves_disagree(data_path, foldername='globalvalues', args=args)
                                waves = None
                            elif args.dataset=='nytimes': 
                                q_IDS = get_q_IDs_waves_disagree(data_path, foldername=demographic_group, args=args)
                                waves = None
                            else: 
                                q_IDS, waves = None, None 
                            
                            task_disagree500(q_IDS, demographic, data_path, model, args, waves, demographic_group, question_type='task{}'.format(str(task_num)), ficticious_group_ablation=args.ficticious_group_ablation, pipeline=pipeline)
                            
if __name__ == "__main__":
    main()