import os
import json
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict


# this function takes in model_probs and reference_probs and returns the rescaled model_probs. 
# e.g., input: model_probs: np.array(lists): each list is a pair of HT tokens [0.95, 0.05]; reference_probs: np.array(lists): each list is a pair of HT tokens [0.9, 0.1]; 

def rescale_prob(model_probs: np.array, reference_probs: np.array) -> np.array:

    def objective(temperature):
        total_tv = 0
        for i, model_prob_pair in enumerate(model_probs): 
            p=model_prob_pair ** (1.0/temperature) #scaled! 
            p=p/np.sum(p) # normalize! 
            q=reference_probs[i] 
            
            total_tv += 0.5*np.sum(np.abs(p-q))
        return total_tv
    res = minimize_scalar(objective, bounds=(0.0, 10), method='bounded')
    print(res.x)
    scaled = []
    for model_p in model_probs:
        scaled_p = model_p**(1.0/res.x)
        scaled.append(scaled_p/np.sum(scaled_p))

    return scaled, res.fun


def main(): 

    dataset_to_demgroup = {'opinionqa': ['POLPARTY', 'SEX', 'RACE'], 'nytimes': ['POLPARTY', 'SEX'], 'globalvalues': ['globalvalues'] }
    dem_group_to_dem_mapping = {'POLPARTY': ['Democrat', 'Republican'],
                                'SEX': ['Male', 'Female'],
                                'RACE': ['Black', 'White'], 
                                'globalvalues': ['0', '1', '2']}


    for model in ['llama3-70b', 'gpt-4', 'gpt-3.5-turbo-0125']:
        for dataset in ['nytimes', 'opinionqa', 'globalvalues']:
            for steering_method in ['task0', 'task1', 'task3_easy_hard']: 
                model_ps, reference_ps = [], [] # compute a temperature for each steering method in each dataset of each model\
                if steering_method=='task0' and dataset=='globalvalues': continue # did not compute task0 for global values
                
                for dem_group in dataset_to_demgroup[dataset]:
                    for dem in dem_group_to_dem_mapping[dem_group]:
                        
                        if steering_method=='task0':
                            dem_group = 'NONE'
                            dem='Democrat'
                            expected_results_str='expected_results_POLPARTY_Democrat'

                        else: expected_results_str='expected_results'
                        
                        if dataset=='opinionqa': wave='Pew_American_Trends_Panel_disagreement_100'
                        else: wave=''

                        #print(steering_method, model, dem_group, dem)

                        file_name = '{}/results/{}/model_logprobs/{}/{}/{}/{}/{}.json'.format(os.getcwd(), dataset,model,steering_method, wave, dem_group, dem)

                        # Read data from a JSON file
                        with open(file_name, 'r') as json_file:
                            dem_data = json.load(json_file)
                        
                        #print(len(dem_data.keys()))
                        for i, qID in enumerate(dem_data): 
                            model_ps.append(list(dem_data[qID]['avg_actual_results'].values()))
                            reference_ps.append(list(dem_data[qID][expected_results_str].values()))

                rescaled_ps, _ = rescale_prob(model_ps, reference_ps)

                count_2=0
                for dem_group in dataset_to_demgroup[dataset]:
                    for dem in dem_group_to_dem_mapping[dem_group]:
                        if steering_method=='task0':
                            dem_group = 'NONE'
                            dem='Democrat'
                            expected_results_str='expected_results_POLPARTY_Democrat'

                        else: expected_results_str='expected_results'
                        if dataset=='opinionqa': wave='Pew_American_Trends_Panel_disagreement_100'
                        else: wave=''

                        file_name = '{}/results/{}/model_logprobs/{}/{}/{}/{}/{}.json'.format(os.getcwd(), dataset,model,steering_method, wave, dem_group, dem)

                        # Read data from a JSON file
                        with open(file_name, 'r') as json_file:
                            dem_data = json.load(json_file)

                        # write a new dictionary that has the rescaled probs
                        for i, qID in enumerate(dem_data):    
                            dem_data[qID]['avg_actual_results'] = dict(zip(list(dem_data[qID]['avg_actual_results'].keys()), rescaled_ps[count_2]))
                            count_2+=1
                        
                        rescaled_filename = file_name.split('model_logprobs')[0]+'rescaled_model_logprobs'+file_name.split('model_logprobs')[1]
                        
                        if not os.path.exists(file_name.split('model_logprobs')[0]+'rescaled_model_logprobs/' + model): os.makedirs(file_name.split('model_logprobs')[0]+'rescaled_model_logprobs/'+model, exist_ok=True)
                        if not os.path.exists(file_name.split('model_logprobs')[0]+'rescaled_model_logprobs/' + model+'/'+steering_method): os.makedirs(file_name.split('model_logprobs')[0]+'rescaled_model_logprobs/'+model+'/'+steering_method, exist_ok=True) 
                        if not os.path.exists(file_name.split('model_logprobs')[0]+'rescaled_model_logprobs/' + model+'/'+steering_method+'/'+wave): os.makedirs(file_name.split('model_logprobs')[0]+'rescaled_model_logprobs/'+model+'/'+steering_method+'/'+wave, exist_ok=True)
                        if not os.path.exists(file_name.split('model_logprobs')[0]+'rescaled_model_logprobs/' + model+'/'+steering_method+'/'+wave+'/'+dem_group): os.makedirs(file_name.split('model_logprobs')[0]+'rescaled_model_logprobs/'+model+'/'+steering_method+'/'+wave+'/'+dem_group, exist_ok=True)
                        
                        with open(rescaled_filename, 'w') as json_file:
                            json.dump(dem_data, json_file)


if __name__ == '__main__': 
    main()