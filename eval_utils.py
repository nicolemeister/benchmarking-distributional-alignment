import os
import json
import numpy as np
import argparse
from utils import *
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


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
    if (task_type=='task0' and output_type=='sequence' and model=='gpt-3.5-turbo-0125') or (task_type=='task1' and output_type=='sequence' and model=='gpt-3.5-turbo-0125' and dataset=='globalvalues'): 
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

# bootstrapping and mean of one dataset
def compute_one(input_data):   
    # print(np.array(input_data).shape)
    data = []

    # convert the list of lists into a full list 
    for lst in input_data:
        data.extend(eval(lst))


    num_bootstraps = 1000

    def compute_statistic(sample):
        # todo: weighted average so that nytimes is weighted 50% and opinionqa is weighted 50%
        try: return np.mean(sample)  
        except: breakpoint()

    # Bootstrapping process
    bootstrap_statistics = []
    for _ in range(num_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        statistic = compute_statistic(bootstrap_sample)
        bootstrap_statistics.append(statistic)

    # 95% confidence interval
    confidence_level = 0.95
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
    upper_bound = np.percentile(bootstrap_statistics, upper_percentile)
    
    return np.mean(bootstrap_statistics), (upper_bound-lower_bound)/2


# bootstrapping and mean of three datasets
def compute_three(opinionqa_data, nytimes_data, globalvalues_data, human_data=False):     
    opinionqa, nytimes, globalvalues = [], [], []   
    

    # convert the list of lists into a full list 
    if not human_data: 
        for lst in opinionqa_data:
            
            try: opinionqa.extend(eval(str(lst)))
            except: breakpoint()
            # except: opinionqa.extend(list(lst))
    else: opinionqa = opinionqa_data

    # convert the list of lists into a full list 
    if not human_data: 
        for lst in nytimes_data:
            try: nytimes.extend(eval(str(lst)))
            except: breakpoint()
            # except: nytimes.extend(list(lst))
    else: nytimes = nytimes_data

    if not human_data: 
        for lst in globalvalues_data:
            try: globalvalues.extend(eval(str(lst)))
            except: breakpoint()
            # except: globalvalues.extend(list(lst))
    else: globalvalues = globalvalues_data


    num_bootstraps = 1000
    def compute_statistic(opinionqa, nytimes, globalvalues):
        
        # Calculate averages
        try: average_opinionqa= np.mean(opinionqa)
        except: print(opinionqa)
        try: average_nytimes = np.mean(nytimes)
        except: print(nytimes)
        try: average_globalvalues = np.mean(globalvalues)
        except: print(globalvalues) 
        weighted_average = (1.0/3.0) * average_opinionqa + (1.0/3.0) * average_nytimes + (1.0/3.0) * average_globalvalues
        return weighted_average

    # Bootstrapping process
    bootstrap_statistics = []
    for _ in range(num_bootstraps):
        bootstrap_sample_oqa = np.random.choice(opinionqa, size=len(opinionqa), replace=True)
        bootstrap_sample_nyt = np.random.choice(nytimes, size=len(nytimes), replace=True)
        bootstrap_sample_gv = np.random.choice(globalvalues, size=len(globalvalues), replace=True)
        statistic = compute_statistic(bootstrap_sample_oqa, bootstrap_sample_nyt, bootstrap_sample_gv)
        bootstrap_statistics.append(statistic)

    # 95% confidence interval
    confidence_level = 0.95
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
    upper_bound = np.percentile(bootstrap_statistics, upper_percentile)
    return np.mean(bootstrap_statistics), (upper_bound-lower_bound)/2


def normalize(array):
    
  total_sum = np.sum(array)
  normalized_array = array / total_sum
  return normalized_array

def compute_tv_human_annotations_NYT(df, nosteer=True, k=4, ingroup=False, outgroup=False, ingroup_name = 'Democrat', qID_to_qualtricsID=None):
  if ingroup: print('in group')
  if outgroup: print('out group')
  dem_group_to_dem_mapping = {
                              'POLPARTY': ['Democrat', 'Republican'],
                              'SEX': ['Male', 'Female']
                              }

  data_path ='/Users/nicolemeister/Desktop/STANFORD/distributions/NYTIMES/NONE_data.json'
  with open(data_path, 'r') as json_file:
    # Load JSON data
    data = json.load(json_file)

  dem_to_qualtrics_map = {'Democrat': 'Q_NoSteering', 'Republican': 'Q135', 'Male':'Q136', 'Female': 'Q137'}
  dem_to_demgroup = {'Democrat': 'POLPARTY', 'Republican': 'POLPARTY', 'Male':'SEX', 'Female': 'SEX'}

  tv_across_groups, hv_across_groups, expected_across_groups = [], [], []

  if ingroup: 
    list_of_groups = [ingroup_name]
  elif outgroup:
    list_of_groups = dem_group_to_dem_mapping[dem_to_demgroup[ingroup_name]].copy()
    list_of_groups.remove(ingroup_name)
  else: list_of_groups = list(dem_to_demgroup.keys())
  # print(ingroup_name, list_of_groups)

  for dem in list_of_groups:
    num_responses, all_tvs = [], []
    for qID in list(data.keys()):
      qualtricsID = qID_to_qualtricsID[qID]
      cols = []
      if nosteer: qualtrics_str = 'Q_NoSteering'
      else: 
        qualtrics_str = dem_to_qualtrics_map[dem]
      cols.append(str(qualtricsID)+'_{}_1'.format(qualtrics_str))
      cols.append(str(qualtricsID)+'_{}_2'.format(qualtrics_str))
      cols.append(str(qualtricsID)+'_{}_3'.format(qualtrics_str))
      cols.append(str(qualtricsID)+'_{}_4'.format(qualtrics_str))

      df_cleaned = df[cols].dropna()[:k]
      num_responses.append(len(df_cleaned))
      if len(df_cleaned)>0: 
        expected_results_str = '{}'.format(dem)
        MC_options = list(data[qID][expected_results_str].keys())
        human_values = np.mean(np.array(df_cleaned, dtype=np.float64), axis=0)[:len(MC_options)] * 0.01
        human_results = dict(zip(MC_options, human_values))

        gt_results = normalize(np.array(list(data[qID][expected_results_str].values())))
        # print(human_values, gt_results)
        tv = calc_total_variation(human_values, gt_results)
        all_tvs.append(tv)
        hv_across_groups.append(human_values)
        expected_across_groups.append(gt_results)
    tv_across_groups.extend(all_tvs)
    mean, bs = compute_one(all_tvs)
    print("{}: {:.3f} +/- {:.3f}".format(dem, mean, bs))
    
    # print(len(all_tvs))
  mean, bs = compute_one(tv_across_groups)
  return mean, bs, tv_across_groups, num_responses# , hv_across_groups, expected_across_groups


def compute_tv_human_annotations_OQA(df, nosteer=True, k=4, ingroup=False, outgroup=False, ingroup_name = 'Democrat', qID_to_qualtricsID=None):
  if ingroup: print('in group')
  if outgroup: print('out group')
  dem_group_to_dem_mapping = {
                              'POLPARTY': ['Democrat', 'Republican'],
                              'SEX': ['Male', 'Female'],
                              'RACE': ['Black', 'White']
                              }

  data_path ='{}/results/opinionqa/express_distribution/gpt-3.5-turbo-0125/task0/Pew_American_Trends_Panel_disagreement_100/NONE/Democrat.json'.format(os.getcwd())
  with open(data_path, 'r') as json_file:
    data = json.load(json_file)   

  dem_to_qualtrics_map = {'Democrat': 'Q_NoSteering', 'Republican': 'Q135', 'Male':'Q136', 'Female': 'Q137', 'Black': 'Q138', 'White': 'Q139'}
  dem_to_demgroup = {'Democrat': 'POLPARTY', 'Republican': 'POLPARTY', 'Male':'SEX', 'Female': 'SEX', 'Black': 'RACE', 'White': 'RACE'}

  tv_across_groups, hv_across_groups, expected_across_groups = [], [], []

  if ingroup: 
    list_of_groups = [ingroup_name]
  elif outgroup:
    list_of_groups = dem_group_to_dem_mapping[dem_to_demgroup[ingroup_name]].copy()
    list_of_groups.remove(ingroup_name)
  else: list_of_groups = list(dem_to_demgroup.keys())
  # print(ingroup_name, list_of_groups)

  for dem in list_of_groups:
    num_responses, all_tvs = [], []
    for qID in list(data.keys()):
      qualtricsID = qID_to_qualtricsID[qID]
      cols = []
      if nosteer: qualtrics_str = 'Q_NoSteering'
      else: 
        qualtrics_str = dem_to_qualtrics_map[dem]
      cols.append(str(qualtricsID)+'_{}_1'.format(qualtrics_str))
      cols.append(str(qualtricsID)+'_{}_2'.format(qualtrics_str))
      cols.append(str(qualtricsID)+'_{}_3'.format(qualtrics_str))
      cols.append(str(qualtricsID)+'_{}_4'.format(qualtrics_str))
      cols.append(str(qualtricsID)+'_{}_5'.format(qualtrics_str))
      cols.append(str(qualtricsID)+'_{}_6'.format(qualtrics_str))
      cols.append(str(qualtricsID)+'_{}_7'.format(qualtrics_str))

      df_cleaned = df[cols].dropna()[:k]
      num_responses.append(len(df_cleaned))
      if len(df_cleaned)>0: 
        expected_results_str = 'expected_results_{}_{}'.format(dem_to_demgroup[dem], dem)
        MC_options = list(data[qID][expected_results_str].keys())
        human_values = np.mean(np.array(df_cleaned, dtype=np.float64), axis=0)[:len(data[qID][expected_results_str].keys())] * 0.01
        human_results = dict(zip(MC_options, human_values))

        gt_results = np.array(list(data[qID][expected_results_str].values()))
        # print(human_values, gt_results)
        tv = calc_total_variation(human_values, gt_results)
        all_tvs.append(tv)
        hv_across_groups.append(human_values)
        expected_across_groups.append(gt_results)
    tv_across_groups.extend(all_tvs)
    mean, bs = compute_one(all_tvs)
    print("{}: {:.3f} +/- {:.3f}".format(dem, mean, bs))

  mean, bs = compute_one(tv_across_groups)
  return mean, bs, tv_across_groups, num_responses#  , hv_across_groups, expected_across_groups


def human_eval_NYT():
   
   col_to_keep = ['QDem_Age', 'QDem_Gend', 'QDem_Race', 'Q_DemRepub', 'QDem_Income'] # demographics

   data_path = '{}/nytimes/NONE_data.json'.format(os.getcwd())
   with open(data_path, 'r') as json_file:
        # Load JSON data
        data = json.load(json_file)

   qualtricsID_to_qID, qID_to_qualtricsID = {}, {}
   for i, qualtricsID in enumerate([1] + list(np.arange(103, 337))):
      col_to_keep.append(str(qualtricsID)+'_Q138')
      col_to_keep.append(str(qualtricsID)+'_Q_NoSteering_1')
      col_to_keep.append(str(qualtricsID)+'_Q_NoSteering_2')
      col_to_keep.append(str(qualtricsID)+'_Q_NoSteering_3')
      col_to_keep.append(str(qualtricsID)+'_Q_NoSteering_4')
      qualtricsID_to_qID[qualtricsID] = list(data.keys())[i]
      qID_to_qualtricsID[list(data.keys())[i]] = qualtricsID
    

   # NO STEER DF
   nosteer_df_1 = pd.read_csv('{}/results/human_annotations/Distributional_Alignment_No_Steering - NYT_July 1, 2024_12.41.csv'.format(os.getcwd()))
   nosteer_df_1 = nosteer_df_1.drop(index=[0, 1])
   
   nosteer_df_2 = pd.read_csv('{}/results/human_annotations/Distributional_Alignment_No_Steering - NYT - V2_July 2, 2024_18.52.csv'.format(os.getcwd()))
   nosteer_df_2 = nosteer_df_2.drop(index=[0, 1])
   
   nosteer_df =  pd.concat([nosteer_df_1, nosteer_df_2])
   
   # PERSONA DF
   persona_df = pd.read_csv('{}/results/human_annotations/Distributional_Alignment_Persona_Steering - NYT_July 3, 2024_10.32.csv'.format(os.getcwd()))
   persona_df = persona_df.drop(index=[0, 1])

   fewshot_df = pd.read_csv('{}/results/human_annotations/Distributional_Alignment_FewShot_Steering - NYT_July 17, 2024_10.52.csv'.format(os.getcwd()))
   fewshot_df = fewshot_df.drop(index=[0, 1])

   return nosteer_df, persona_df, fewshot_df, qID_to_qualtricsID

def human_eval_OQA():
   nosteer_df = pd.read_csv('/Users/nicolemeister/Desktop/STANFORD/distributions/results/human_annotations/Distributional_Alignment_No_Steering_May 21, 2024_18.01.csv')
   persona_df = pd.read_csv('/Users/nicolemeister/Desktop/STANFORD/distributions/results/human_annotations/Distributional_Alignment_Persona_Steering_May 21, 2024_19.06.csv')
   fewshot_df = pd.read_csv('/Users/nicolemeister/Desktop/STANFORD/distributions/results/human_annotations/Distributional_Alignment_FewShot_Steering_May 21, 2024_18.00.csv')

   nosteer_df = nosteer_df.drop(index=[0, 1])
   persona_df = persona_df.drop(index=[0, 1])
   fewshot_df = fewshot_df.drop(index=[0, 1])

   col_to_keep = ['QDem_Age', 'QDem_Gend', 'QDem_Race', 'Q_DemRepub', 'QDem_Income'] # demographics

   data_path ='{}/results/opinionqa/express_distribution/gpt-3.5-turbo-0125/task0/Pew_American_Trends_Panel_disagreement_100/NONE/Democrat.json'.format(os.getcwd())
   with open(data_path, 'r') as json_file:
      # Load JSON data
      data = json.load(json_file)
      
   qualtricsID_to_qID, qID_to_qualtricsID = {}, {}
   for i, qualtricsID in enumerate([1] + list(np.arange(4, 103))):
      col_to_keep.append(str(qualtricsID)+'_Q138')
      col_to_keep.append(str(qualtricsID)+'_Q_NoSteering_1')
      col_to_keep.append(str(qualtricsID)+'_Q_NoSteering_2')
      col_to_keep.append(str(qualtricsID)+'_Q_NoSteering_3')
      col_to_keep.append(str(qualtricsID)+'_Q_NoSteering_4')
      col_to_keep.append(str(qualtricsID)+'_Q_NoSteering_5')
      col_to_keep.append(str(qualtricsID)+'_Q_NoSteering_6')
      col_to_keep.append(str(qualtricsID)+'_Q_NoSteering_7')
      qualtricsID_to_qID[qualtricsID] = list(data.keys())[i]
      qID_to_qualtricsID[list(data.keys())[i]] = qualtricsID    
    

   return nosteer_df, persona_df, fewshot_df, qID_to_qualtricsID

def add_humanevaldata_to_df(df, dataset, task_type, human_annotation_data, model='human', output_type='express_distribution', wave = None, demographic_group='all', demographic='all'):
    has_values = ((df['Task Type']==task_type) & (df['Model']==model) & (df['Dataset']==dataset) & (df['Wave']==wave) & (df['Demographic Group/Avg']==demographic_group) & (df['Demographic/Avg']==demographic) & (df['Output Type']==output_type)).any()
    if not has_values: 
        
        # Define data for the new row
        new_row = {'Task Type': task_type, 'Model': model, 'Dataset':dataset, 'Wave': wave, 'Demographic Group/Avg': demographic_group, 'Demographic/Avg': demographic, 'Output Type': output_type, 'TV': "{:.5f}".format(human_annotation_data[task_type]['mean']), 'all_tvs': human_annotation_data[task_type]['data']}
        # Append the new row to the DataFrame
        # Create a DataFrame with the new row
        new_df = pd.DataFrame([new_row])

        # Concatenate the new DataFrame with the original DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
    return df
   

def human_eval(eval_result_df_path):

    try: df = pd.read_csv(eval_result_df_path)
    except: 
       # error message
       print('Error: eval_result_df_path does not exist. run the model eval code first! (lm_steering_eval.py)')
       return 

    # NYT
    nosteer_df, persona_df, fewshot_df, qID_to_qualtricsID = human_eval_NYT()

    nosteer_mean, nosteer_bs, nosteer_data, _ = compute_tv_human_annotations_NYT(nosteer_df, nosteer=True, ingroup=False, outgroup=False, qID_to_qualtricsID=qID_to_qualtricsID)
    persona_mean, persona_bs, persona_data, _ = compute_tv_human_annotations_NYT(persona_df, nosteer=False, ingroup=False, outgroup=False, qID_to_qualtricsID=qID_to_qualtricsID)
    fewshot_mean, fewshot_bs, fewshot_data, _ = compute_tv_human_annotations_NYT(fewshot_df, nosteer=False, ingroup=False, outgroup=False, qID_to_qualtricsID=qID_to_qualtricsID)

    print("No Steer: {:4f} +/- {:4f}".format(nosteer_mean, nosteer_bs))
    print("Persona: {:4f} +/- {:4f}".format(persona_mean, persona_bs))
    print("Few Shot: {:4f} +/- {:4f}".format(fewshot_mean, fewshot_bs))

    NYT_human_annotation_data = {'task0': {'data': nosteer_data, 'mean': nosteer_mean, 'bs': nosteer_bs}, 'task1': {'data': persona_data, 'mean': persona_mean, 'bs': persona_bs}, 'task3_easy_hard': {'data': fewshot_data, 'mean': fewshot_mean, 'bs': fewshot_bs}}

    # add to df 

    for task_type in ['task0', 'task1', 'task3_easy_hard']:
       dataset = 'nytimes'
       df = add_humanevaldata_to_df(df, dataset, task_type, NYT_human_annotation_data)

    df.to_csv(eval_result_df_path, index=False)

    # OQA
    nosteer_df, persona_df, fewshot_df, qID_to_qualtricsID = human_eval_OQA()

    nosteer_mean, nosteer_bs, nosteer_data, _ = compute_tv_human_annotations_OQA(nosteer_df, nosteer=True, ingroup=False, outgroup=False, qID_to_qualtricsID=qID_to_qualtricsID)
    persona_mean, persona_bs, persona_data, _ = compute_tv_human_annotations_OQA(persona_df, nosteer=False, ingroup=False, outgroup=False, qID_to_qualtricsID=qID_to_qualtricsID)
    fewshot_mean, fewshot_bs, fewshot_data, _ = compute_tv_human_annotations_OQA(fewshot_df, nosteer=False, ingroup=False, outgroup=False, qID_to_qualtricsID=qID_to_qualtricsID)

    print("No Steer: {:4f} +/- {:4f}".format(nosteer_mean, nosteer_bs))
    print("Persona: {:4f} +/- {:4f}".format(persona_mean, persona_bs))
    print("Few Shot: {:4f} +/- {:4f}".format(fewshot_mean, fewshot_bs))

    OQA_human_annotation_data = {'task0': {'data': nosteer_data, 'mean': nosteer_mean, 'bs': nosteer_bs}, 'task1': {'data': persona_data, 'mean': persona_mean, 'bs': persona_bs}, 'task3_easy_hard': {'data': fewshot_data, 'mean': fewshot_mean, 'bs': fewshot_bs}}

    # add to df 
    for task_type in ['task0', 'task1', 'task3_easy_hard']:
       dataset = 'opinionqa'
       df = add_humanevaldata_to_df(df, dataset, task_type, OQA_human_annotation_data)
    df.to_csv(eval_result_df_path, index=False)
    
    return 


def compute_both(opinionqa_data, nytimes_data, human_data=False):     
    opinionqa, nytimes = [], []
    

    # convert the list of lists into a full list 
    if not human_data: 
        for lst in opinionqa_data:
            opinionqa.extend(eval(lst))
            # except: opinionqa.extend(list(lst))
    else: opinionqa = opinionqa_data

    # convert the list of lists into a full list 
    if not human_data: 
        for lst in nytimes_data:
            nytimes.extend(eval(lst))
            # except: nytimes.extend(list(lst))
    else: nytimes = nytimes_data

    print(len(opinionqa), len(nytimes))
    num_bootstraps = 1000
    def compute_statistic(opinionqa, nytimes):
        
        # Calculate averages
        try: average_opinionqa= np.mean(opinionqa)
        except: print(opinionqa)
        try: average_nytimes = np.mean(nytimes)
        except: print(nytimes)
        weighted_average = 0.5 * average_opinionqa + 0.5 * average_nytimes
        return weighted_average

    # Bootstrapping process
    bootstrap_statistics = []
    for _ in range(num_bootstraps):
        bootstrap_sample_oqa = np.random.choice(opinionqa, size=len(opinionqa), replace=True)
        bootstrap_sample_nyt = np.random.choice(nytimes, size=len(nytimes), replace=True)
        statistic = compute_statistic(bootstrap_sample_oqa, bootstrap_sample_nyt)
        bootstrap_statistics.append(statistic)

    # 95% confidence interval
    confidence_level = 0.95
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
    upper_bound = np.percentile(bootstrap_statistics, upper_percentile)
    return np.mean(bootstrap_statistics), (upper_bound-lower_bound)/2


def compute_K2S_gap(distrib_alignment_leaderboard_path, leaderboard_path):
    distrib_alignment_df=pd.read_csv(distrib_alignment_leaderboard_path)
    
    if os.path.exists(leaderboard_path):
        leaderboard_df = pd.read_csv(leaderboard_path)
    else: leaderboard_df = pd.DataFrame(columns=['Model Name', 'Dataset', 'Task Type', 'K2S Mean'])

    models = ['gpt-3.5-turbo-0125', 'gpt-4', 'anthropic_haiku', 'anthropic_opus', 'llama3-70b']

    for model in models: 
        print(model)
        verb = distrib_alignment_df[(distrib_alignment_df['Model Name']=="{} {}".format(model, "(V)"))]['Alignment Mean'].iloc[0]
        seq = distrib_alignment_df[(distrib_alignment_df['Model Name']=="{} {}".format(model, "(Seq)"))]['Alignment Mean'].iloc[0]
        KS_gap = ((seq/verb)-1)
        # Adding a data entry
        new_entry = pd.DataFrame({
            'Model Name': [model],
            'Dataset': ['three'],
            'Task Type': ['knowledge_to_sim_gap'],
            'K2S Mean': [KS_gap],
        })

        # Adding the new entry to the dataframe
        leaderboard_df = pd.concat([leaderboard_df, new_entry], ignore_index=True)

    leaderboard_df.to_csv(leaderboard_path, index=False)
    return 





def distrib_alignment_leaderboard(eval_result_df_path, leaderboard_path):

    df=pd.read_csv(eval_result_df_path)
    if os.path.exists(leaderboard_path):
        leaderboard_df = pd.read_csv(leaderboard_path)
    else: leaderboard_df = pd.DataFrame(columns=['Model Name', 'Dataset', 'Task Type', 'Alignment Mean', 'Alignment Error'])
    

    output_type_to_latex = {'express_distribution': "V", 'sequence': "Seq", 'model_logprobs': "Log-p", 'rescaled_model_logprobs': "TS-Log-p"}

    models = ['gpt-3.5-turbo-0125', 'gpt-4', 'anthropic_haiku', 'anthropic_opus', 'llama3-70b']
    output_types = ['express_distribution', 'sequence', 'model_logprobs', 'rescaled_model_logprobs']

    for model in models: 
        print(model)
        for output_type in output_types: 
            print(output_type)
            
            opinionqa_data = df[(df.Model==model) & (df['Output Type']==output_type) & (df['Dataset']=='opinionqa') & ((df['Task Type']=='task1') | (df['Task Type']=='task3_easy_hard'))]['all_tvs']
            nytimes_data = df[(df.Model==model) & (df['Output Type']==output_type) & (df['Dataset']=='nytimes') & ((df['Task Type']=='task1') | (df['Task Type']=='task3_easy_hard'))]['all_tvs']
            globalvalues_data = df[(df.Model==model) & (df['Output Type']==output_type) & (df['Dataset']=='globalvalues') & ((df['Task Type']=='task1') | (df['Task Type']=='task3_easy_hard'))]['all_tvs']
            mean, bs = compute_three(opinionqa_data, nytimes_data, globalvalues_data) 

            # Adding a data entry
            new_entry = pd.DataFrame({
                'Model Name': ["{} ({})".format(model, output_type_to_latex[output_type])],
                'Dataset': ['three'],
                'Task Type': ['task1_task3'],
                'Alignment Mean': [mean],
                'Alignment Error': [bs]
            })

            # Adding the new entry to the dataframe
            leaderboard_df = pd.concat([leaderboard_df, new_entry], ignore_index=True)

            print('{:.4f} +/- {:.4f}'.format(mean, bs))

        leaderboard_df.to_csv(leaderboard_path, index=False)


    task_types = ['ground_truth', 'uniform', 'LB1', 'LB2']
    # output_types = ['express_distribution', 'sequence']
    for task_type in task_types:

        opinionqa_data = df[(df.Model=='simulated') & (df['Task Type']==task_type) & (df['Dataset']=='opinionqa')]['all_tvs']
        nytimes_data = df[(df.Model=='simulated') & (df['Task Type']==task_type) & (df['Dataset']=='nytimes')]['all_tvs']
        globalvalues_data = df[(df.Model=='simulated') & (df['Task Type']==task_type) & (df['Dataset']=='globalvalues')]['all_tvs']
        mean, bs = compute_three(opinionqa_data, nytimes_data, globalvalues_data)

        # Adding a data entry
        new_entry = pd.DataFrame({
            'Model Name': ['simulated'],
            'Dataset': ['three'],
            'Task Type': [task_type],
            'Alignment Mean': [mean],
            'Alignment Error': [bs]
        })

        # Adding the new entry to the dataframe
        leaderboard_df = pd.concat([leaderboard_df, new_entry], ignore_index=True)

        print('{:.4f} +/- {:.4f}'.format(mean, bs))
    
    leaderboard_df.to_csv(leaderboard_path, index=False)

    return 

# TO DO: FIX THIS! 
def distrib_alignment_leaderboard_all(eval_result_df_path, leaderboard_path):

    df=pd.read_csv(eval_result_df_path)
    if os.path.exists(leaderboard_path):
        leaderboard_df = pd.read_csv(leaderboard_path)
    else: leaderboard_df = pd.DataFrame(columns=['Model Name', 'Dataset', 'Task Type', 'Alignment Mean', 'Alignment Error'])
    

    output_type_to_latex = {'express_distribution': "V", 'sequence': "Seq", 'model_logprobs': "Log-p", 'rescaled_model_logprobs': "TS-Log-p"}

    models = ['gpt-3.5-turbo-0125', 'gpt-4', 'anthropic_haiku', 'anthropic_opus', 'llama3-70b']
    output_types = ['express_distribution', 'sequence', 'model_logprobs', 'rescaled_model_logprobs']
    task_types = ['task0', 'task1','task3_easy_hard']

    for dataset in ['opinionqa', 'nytimes', 'globalvalues']:
        for task_type in task_types:
            print(task_type)
            for model in models: 
                print(model)
                for output_type in output_types: 
                    print(output_type)
                    
                    input_data = df[(df.Model==model) & (df['Output Type']==output_type) & (df['Dataset']==dataset) & (df['Task Type']==task_type)]['all_tvs']
                    mean, bs = compute_one(input_data)

                    # Adding a data entry
                    new_entry = pd.DataFrame({
                        'Model Name': ["{} ({})".format(model, output_type_to_latex[output_type])],
                        'Dataset': [dataset],
                        'Task Type': [task_type],
                        'Alignment Mean': [mean],
                        'Alignment Error': [bs]
                    })

                    # Adding the new entry to the dataframe
                    leaderboard_df = pd.concat([leaderboard_df, new_entry], ignore_index=True)

                    print('{:.4f} +/- {:.4f}'.format(mean, bs))
    
    # simulated values
    task_types = ['ground_truth', 'uniform', 'LB1', "LB2"]
    for task_type in task_types:
        for dataset in ['opinionqa', 'nytimes', 'globalvalues']:
            print(task_type)
            input_data = df[(df.Model=='simulated') & (df['Task Type']==task_type) & (df['Dataset']==dataset)]['all_tvs']
            mean, bs = compute_one(input_data)

            # Adding a data entry
            new_entry = pd.DataFrame({
                'Model Name': ['simulated'],
                'Dataset': [dataset],
                'Task Type': [task_type],
                'Alignment Mean': [mean],
                'Alignment Error': [bs]
            })

            # Adding the new entry to the dataframe
            leaderboard_df = pd.concat([leaderboard_df, new_entry], ignore_index=True)

            print('{:.4f} +/- {:.4f}'.format(mean, bs))



    # compute avg of OQA and NYT for the human results leaderboard 


    for model in models: 
        print(model)
        for output_type in output_types: 
            print(output_type)
            
            opinionqa_data = df[(df.Model==model) & (df['Output Type']==output_type) & (df['Dataset']=='opinionqa') & ((df['Task Type']=='task1') | (df['Task Type']=='task3_easy_hard'))]['all_tvs']
            nytimes_data = df[(df.Model==model) & (df['Output Type']==output_type) & (df['Dataset']=='nytimes') & ((df['Task Type']=='task1') | (df['Task Type']=='task3_easy_hard'))]['all_tvs']
            mean, bs = compute_both(opinionqa_data, nytimes_data)

            # Adding a data entry
            new_entry = pd.DataFrame({
                'Model Name': ["{} ({})".format(model, output_type_to_latex[output_type])],
                'Dataset': ['both'],
                'Task Type': ['task1_task3'],
                'Alignment Mean': [mean],
                'Alignment Error': [bs]
            })

            # Adding the new entry to the dataframe
            leaderboard_df = pd.concat([leaderboard_df, new_entry], ignore_index=True)

            print('{:.4f} +/- {:.4f}'.format(mean, bs))

    leaderboard_df.to_csv(leaderboard_path, index=False)


    # human eval
    opinionqa_data = df[(df.Model=='human') & (df['Dataset']=='opinionqa') & ((df['Task Type']=='task1') | (df['Task Type']=='task3_easy_hard'))]['all_tvs']
    nytimes_data = df[(df.Model=='human') & (df['Dataset']=='nytimes') & ((df['Task Type']=='task1') | (df['Task Type']=='task3_easy_hard'))]['all_tvs']
    mean, bs = compute_both(opinionqa_data, nytimes_data)

    # Adding a data entry
    new_entry = pd.DataFrame({
        'Model Name': ['human'],
        'Dataset': ['both'],
        'Task Type': ['task1_task3'],
        'Alignment Mean': [mean],
        'Alignment Error': [bs]})

    # Adding the new entry to the dataframe
    leaderboard_df = pd.concat([leaderboard_df, new_entry], ignore_index=True)

    print('{:.4f} +/- {:.4f}'.format(mean, bs))

    leaderboard_df.to_csv(leaderboard_path, index=False)

    return 