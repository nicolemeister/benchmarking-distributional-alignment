import os
import json
import numpy as np
import argparse
from utils import *
from collections import defaultdict
import pandas as pd

def eval_metrics(df, task_type, model, demographic_group, demographic, wave, output_type, dataset):
    # Check if it's already in the dataframe
    has_values = ((df['Task Type']==task_type) & (df['Model']==model) & (df['Dataset']==dataset) & (df['Wave']==wave) & (df['Demographic Group/Avg']==demographic_group) & (df['Demographic/Avg']==demographic) & (df['Output Type']==output_type)).any()
    if not has_values: 
        TV, _, _, all_tvs, _ = compute_tv(task=task_type, model=model, demographic_group=demographic_group, dataset=dataset, demographic=demographic, wave=wave, output_type=output_type)
        
        # Define data for the new row
        new_row = {'Task Type': task_type, 'Model': model, 'Dataset':dataset, 'Wave': wave, 'Demographic Group/Avg': demographic_group, 'Demographic/Avg': demographic, 'Output Type': output_type, 'TV': "{:.5f}".format(TV), 'all_tvs': list(all_tvs)}
        # Append the new row to the DataFrame
        # Create a DataFrame with the new row
        new_df = pd.DataFrame([new_row])

        # Concatenate the new DataFrame with the original DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        

    # compute discretization error and uniform
    if task_type=='task1' and output_type=='sequence' and model=='gpt-3.5-turbo-0125': 
        TV, _, _, all_tvs = compute_tv_GT(task=task_type, model=model, demographic_group=demographic_group, dataset=dataset,  demographic=demographic, wave=wave, output_type=output_type)
        # Define data for the new row
        new_row = {'Task Type': 'ground_truth', 'Model': 'simulated', 'Dataset':dataset, 'Wave': wave, 'Demographic Group/Avg': demographic_group, 'Demographic/Avg': demographic, 'Output Type': output_type, 'TV': "{:.5f}".format(TV),'all_tvs': list(all_tvs)}
        new_df = pd.DataFrame([new_row])

        # Concatenate the new DataFrame with the original DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        
        TV, _, _, all_tvs, _ = compute_tv(task=task_type, model=model, demographic_group=demographic_group, dataset=dataset, demographic=demographic, wave=wave, output_type=output_type, uniform=True)

        # Define data for the new row
        new_row = {'Task Type': 'uniform', 'Model': 'simulated', 'Dataset':dataset, 'Wave': wave, 'Demographic Group/Avg': demographic_group, 'Demographic/Avg': demographic, 'Output Type': output_type, 'TV': "{:.5f}".format(TV), 'all_tvs': list(all_tvs)}
        new_df = pd.DataFrame([new_row])

        # Concatenate the new DataFrame with the original DataFrame
        df = pd.concat([df, new_df], ignore_index=True)

        TV, _, _, all_tvs, _ = compute_tv(task=task_type, model=model, demographic_group=demographic_group, demographic=demographic, wave=wave, output_type=output_type, LB1=True, dataset=dataset)

        # Define data for the new row
        new_row = {'Task Type': 'LB1', 'Model': 'simulated', 'Dataset':dataset, 'Wave': wave, 'Demographic Group/Avg': demographic_group, 'Demographic/Avg': demographic, 'Output Type': output_type, 'TV': "{:.5f}".format(TV), 'all_tvs': list(all_tvs)}
        new_df = pd.DataFrame([new_row])

        # Concatenate the new DataFrame with the original DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        

        TV, _, _, all_tvs, _ = compute_tv(task=task_type, model=model, demographic_group=demographic_group, demographic=demographic, wave=wave, output_type=output_type, LB2=True, dataset=dataset)

        # Define data for the new row
        new_row = {'Task Type': 'LB2', 'Model': 'simulated', 'Dataset':dataset, 'Wave': wave, 'Demographic Group/Avg': demographic_group, 'Demographic/Avg': demographic, 'Output Type': output_type, 'TV': "{:.5f}".format(TV), 'all_tvs': list(all_tvs)}
        new_df = pd.DataFrame([new_row])

        # Concatenate the new DataFrame with the original DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        
    return df 

def main():
    # Define column names
    columns = ['Task Type', 'Model', 'Dataset', 'Wave', 'Demographic Group/Avg', 'Demographic/Avg', 'Output Type', 'TV']
    outputtype_to_model = {'express_distribution': ['gpt-3.5-turbo-0125', 'anthropic_haiku', 'anthropic_opus', 'gpt-4', 'llama3-70b'], 'sequence': ['gpt-3.5-turbo-0125', 'anthropic_haiku', 'anthropic_opus', 'gpt-4', 'llama3-70b'], 'model_logprobs': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b'],'rescaled_model_logprobs': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b'], 
    'rescaled_model_logprobs_T_0_5-per_dataset': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b'], 'rescaled_model_logprobs_T_0_10-per_dataset': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b'], 'rescaled_model_logprobs_T_0_50-per_dataset': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b'], 'rescaled_model_logprobs_T_0_200-per_dataset': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b'], 'rescaled_model_logprobs_T_0_5-per_steering_method': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b'], 'rescaled_model_logprobs_T_0_10-per_steering_method': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b'], 'rescaled_model_logprobs_T_0_50-per_steering_method': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b'], 'rescaled_model_logprobs_T_0_200-per_steering_method': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b']}

    
    eval_result_df_path = '{}/results/eval_disagreement_bootstrapping.csv'.format(os.getcwd())
    if not os.path.exists(eval_result_df_path): 
        df = pd.DataFrame(columns=columns)
    else: df = pd.read_csv(eval_result_df_path)

    dem_group_to_dem_mapping = {'globalvalues': ['0', '1', '2']}

    
    dem_group_to_dem_mapping = {
                            # 'NONE': ['Democrat'], 
                            'POLPARTY': ['Democrat', 'Republican'],
                            'SEX': ['Male', 'Female'],
                            'RACE': ['Black', 'White']
                            }



    for dataset in ['opinionqa', 'nytimes', 'globalvalues']:    
        
        results_path = '/nlp/scr/nmeist/opinion_distributions/results/{}/'.format(dataset)
        for output_type in ['express_distribution', 'sequence', 'model_logprobs', 'rescaled_model_logprobs']: 
            for model in outputtype_to_model[output_type]:
                if model=='.DS_Store': continue
                for task_type in ['task0', 'task1', 'task3_easy_hard']:
                    for wave in os.listdir('{}/{}/{}/{}'.format(results_path, output_type, model, task_type)):
                        for demographic_group in dem_group_to_dem_mapping.keys():
                                for demographic in dem_group_to_dem_mapping[demographic_group]:
                                    print(output_type, model, task_type, wave, demographic_group, demographic)
                                    df=eval_metrics(df, task_type, model, demographic_group, demographic, wave, output_type, dataset)
                        df.to_csv(eval_result_df_path, index=False)
            # Save the DataFrame to a CSV file
            df.to_csv(eval_result_df_path, index=False)

# compute averages
if __name__ == "__main__":
    main()