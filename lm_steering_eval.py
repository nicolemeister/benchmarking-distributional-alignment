import os
import json
import numpy as np
import argparse
from utils import *
from collections import defaultdict
import pandas as pd
import eval_utils

def main():

   
    # Define column names
    columns = ['Task Type', 'Model', 'Dataset', 'Wave', 'Demographic Group/Avg', 'Demographic/Avg', 'Output Type', 'TV']
    outputtype_to_model = {'express_distribution': ['gpt-3.5-turbo-0125', 'anthropic_haiku', 'anthropic_opus', 'gpt-4', 'llama3-70b'], 'sequence': ['gpt-3.5-turbo-0125', 'anthropic_haiku', 'anthropic_opus', 'gpt-4', 'llama3-70b'], 'model_logprobs': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b'],'rescaled_model_logprobs': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b']}
    

    # Take in results from ./results and compute the TV for each task, model, output type, wave, demographic group, demographic. 
    # Store it in this csv file. 
    eval_result_df_path = '{}/results/eval_disagreement_bootstrapping.csv'.format(os.getcwd())
    if not os.path.exists(eval_result_df_path): 
        df = pd.DataFrame(columns=columns)
    else: df = pd.read_csv(eval_result_df_path)

    '''
    # 1. MODEL EVAL 
    dem_group_to_dem_mapping_dataset = {'opinionqa': {'POLPARTY': ['Democrat', 'Republican'], 'SEX': ['Male', 'Female'], 'RACE': ['Black', 'White']}, 
                                        'nytimes': {'POLPARTY': ['Democrat', 'Republican'], 'SEX': ['Male', 'Female']}, 
                                        'globalvalues': {'globalvalues': ['0', '1', '2']}}
    

    for dataset in ['opinionqa', 'nytimes', 'globalvalues']:    
        dem_group_to_dem_mapping = dem_group_to_dem_mapping_dataset[dataset]
        results_path = '{}/results/{}/'.format(os.getcwd(), dataset)
        for output_type in ['express_distribution', 'sequence', 'model_logprobs', 'rescaled_model_logprobs']: 
            for model in outputtype_to_model[output_type]:
                for task_type in ['task0', 'task1', 'task3_easy_hard']:
                    if task_type=='task0' and dataset=='globalvalues': continue
                    for wave in os.listdir('{}/{}/{}/{}'.format(results_path, output_type, model, task_type)):
                        for demographic_group in dem_group_to_dem_mapping.keys():
                                for demographic in dem_group_to_dem_mapping[demographic_group]:
                                    print(output_type, model, task_type, wave, demographic_group, demographic)
                                    df=eval_utils.eval_metrics(df, task_type, model, demographic_group, demographic, wave, output_type, dataset)
                        df.to_csv(eval_result_df_path, index=False)
            
            # Save the DataFrame to a CSV file
            df.to_csv(eval_result_df_path, index=False)

    '''


    '''
    # 2. Human Eval 
    eval_utils.human_eval(eval_result_df_path)
    '''

    # 3. Generate Leaderboard 1: Distributional Alignment Task
    

    # 4. Generate Leaderboard 2: Knowledge to Simulation Gap



if __name__ == "__main__":
    main()