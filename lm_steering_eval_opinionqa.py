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
        TV, _, _, all_tvs, _ = compute_tv(task=task_type, model=model, demographic_group=demographic_group, demographic=demographic, wave=wave, output_type=output_type)
        
        # Define data for the new row
        new_row = {'Task Type': task_type, 'Model': model, 'Dataset':dataset, 'Wave': wave, 'Demographic Group/Avg': demographic_group, 'Demographic/Avg': demographic, 'Output Type': output_type, 'TV': "{:.5f}".format(TV), 'all_tvs': list(all_tvs)}
        # Append the new row to the DataFrame
        # Create a DataFrame with the new row
        new_df = pd.DataFrame([new_row])

        # Concatenate the new DataFrame with the original DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        

    # compute discretization error and uniform
    if task_type=='task0' and output_type=='sequence' and model=='gpt-3.5-turbo-0125' and wave=='Pew_American_Trends_Panel_disagreement_100': 
        TV, _, _, all_tvs = compute_tv_GT(task=task_type, model=model, demographic_group=demographic_group, demographic=demographic, wave=wave, output_type=output_type)
        # Define data for the new row
        new_row = {'Task Type': 'ground_truth', 'Model': 'simulated', 'Dataset':dataset, 'Wave': wave, 'Demographic Group/Avg': demographic_group, 'Demographic/Avg': demographic, 'Output Type': output_type, 'TV': "{:.5f}".format(TV),'all_tvs': list(all_tvs)}
        new_df = pd.DataFrame([new_row])

        # Concatenate the new DataFrame with the original DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        
        TV, _, _, all_tvs, _ = compute_tv(task=task_type, model=model, demographic_group=demographic_group, demographic=demographic, wave=wave, output_type=output_type, uniform=True)

        # Define data for the new row
        new_row = {'Task Type': 'uniform', 'Model': 'simulated', 'Dataset':dataset, 'Wave': wave, 'Demographic Group/Avg': demographic_group, 'Demographic/Avg': demographic, 'Output Type': output_type, 'TV': "{:.5f}".format(TV), 'all_tvs': list(all_tvs)}
        new_df = pd.DataFrame([new_row])

        # Concatenate the new DataFrame with the original DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        
    return df 

def main():
    # Define column names
    columns = ['Task Type', 'Model', 'Dataset', 'Wave', 'Demographic Group/Avg', 'Demographic/Avg', 'Output Type', 'TV']
    outputtype_to_model = {'express_distribution': ['gpt-3.5-turbo-0125', 'anthropic_haiku', 'anthropic_opus', 'gpt-4', 'llama3-70b'], 'sequence': ['gpt-3.5-turbo-0125', 'anthropic_haiku', 'anthropic_opus', 'gpt-4', 'llama3-70b'], 'model_logprobs': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b'],'rescaled_model_logprobs': ['gpt-3.5-turbo-0125', 'gpt-4', 'llama3-70b']}

    # Create an empty DataFrame with specified columns
    # eval_result_df_path = '/nlp/scr/nmeist/opinion_distributions/results/eval_disagreement.csv'
    # eval_result_df_path = '/nlp/scr/nmeist/opinion_distributions/results/eval_disagreement_bootstrapping_final.csv'
    eval_result_df_path = '/nlp/scr/nmeist/opinion_distributions/results/eval_disagreement_bootstrapping_final_v3.csv'
    
    if not os.path.exists(eval_result_df_path): 
        df = pd.DataFrame(columns=columns)
    else: df = pd.read_csv(eval_result_df_path)

    dem_group_to_dem_mapping = {
                            # 'NONE': ['Democrat'], 
                            'POLPARTY': ['Democrat', 'Republican'],
                            'SEX': ['Male', 'Female'],
                            'RACE': ['Black', 'White']
                            }

    dataset = 'opinionqa'
    results_path = '/nlp/scr/nmeist/opinion_distributions/results/{}/'.format(dataset)
    # for output_type in ['express_distribution', 'sequence', 'model_logprobs']: # could change this later to read the path 
    # for output_type in ['express_distribution', 'sequence', 'model_logprobs']: # could change this later to read the path 
    # for output_type in ['express_distribution', 'sequence', 'model_logprobs', 'rescaled_model_logprobs']: 
    for output_type in ['rescaled_model_logprobs']: 
        for model in outputtype_to_model[output_type]:
        # for model in ['llama3-70b', 'gpt-4', 'gpt-3.5-turbo-0125','anthropic_haiku', 'anthropic_opus',]:
        # for model in os.listdir('/nlp/scr/nmeist/opinion_distributions/results/opinionqa/{}'.format(output_type)):
            if model=='.DS_Store': continue
            for task_type in os.listdir('/nlp/scr/nmeist/opinion_distributions/results/opinionqa/{}/{}'.format(output_type, model)):
                if task_type in ['task0', 'task1', 'task3_easy_hard', 'task5']:
                    if task_type=='.DS_Store': continue
                    # for wave in os.listdir('/nlp/scr/nmeist/opinion_distributions/results/opinionqa/{}/{}/{}'.format(output_type, model, task_type)):
                    for wave in os.listdir('{}/{}/{}/{}'.format(results_path, output_type, model, task_type)):
                        if wave=='.DS_Store': continue
                        # do something different for task0: calculate the expected result for each demographic group
                        if task_type=='task0': 
                            for demographic_group in dem_group_to_dem_mapping.keys():
                                for demographic in dem_group_to_dem_mapping[demographic_group]:
                                    df=eval_metrics(df, task_type, model, demographic_group, demographic, wave, output_type, dataset)
                        else: 
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