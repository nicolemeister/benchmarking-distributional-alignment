
import json
import openai
import os
import pandas as pd
import numpy as np
import ast
from collections import defaultdict
import random
import re
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import anthropic
from openai import OpenAI

# from transformers import AutoTokenizer
# import transformers
# import bitsandbytes
# import accelerate
import torch




options={0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R"}
option_to_num ={v: k for k, v in options.items()}

output_type_to_color = {'sequence': 'blue', 'model_logprobs': 'green', 'incontext_avg': 'purple'}

dem_to_color = {'Democrat': 'blue', 'Independent': 'green', 'Republican': 'purple', 'Male': 'blue', 'Female': 'green'}

dem_group_to_dem_mapping = {'NONE': ['Democrat'], 
                            'POLPARTY': ['Democrat', 'Republican'],
                            'SEX': ['Male', 'Female'],
                            'RACE': ['Black', 'White'] ,
                            'globalvalues': ['0', '1', '2'] 
                            }

ficticious_group_ablation_mapping = {'Democrat': 'Foo', 'Independent': 'Bar', 'Republican': 'Baz', 'Male': 'Foo', 'Female': 'Bar'}

all_waves = ['American_Trends_Panel_W29', 'American_Trends_Panel_W27', 'American_Trends_Panel_W42', 'American_Trends_Panel_W45', 'American_Trends_Panel_W26', 'American_Trends_Panel_W43', 'American_Trends_Panel_W50', 'American_Trends_Panel_W92', 'American_Trends_Panel_W32', 'American_Trends_Panel_W34', 'American_Trends_Panel_W41', 'American_Trends_Panel_W49', 'American_Trends_Panel_W82', 'American_Trends_Panel_W36', 'American_Trends_Panel_W54']


def get_results(path, filename, question_category='toy_problems'):
    # Opening JSON file
    f = open('inputs/'+question_category+'/'+path+'/'+ filename+ '.json')
    # returns JSON object as  a dictionary
    data = json.load(f)
    return data

def get_result_format(question_type):
    if question_type=='biased_coin':
        result_format = {'h': 0, 't': 0} 
    else:     
        result_format = {'a': 0, 'b': 0, 'c': 0}
    return result_format

def read_file_to_string(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print("File not found.")
        return None

def calculate_propotions_coin(sequence):
    count_h = sequence.count('H') + sequence.count('h') 
    count_t = sequence.count('T') + sequence.count('t')
    # Calculate proportions
    total_length = len(sequence)

    if (count_h + count_t) < 29:
        print("H Count: {} \t T count: {}".format(count_h, count_t)) 
        return None, None
    else:
        proportion_h = count_h / total_length
        proportion_t = count_t / total_length
        return proportion_h, proportion_t


def get_prompt(question_type):
    if question_type == 'biased_coin' or 'biased_coin_seq' or 'biased_coin_seq_flipped':
        # prompt = 'I will specify a distribution for a biased coin. Please simulate a coin flip just as the probabilities for this biased coin are specified. As an example, if a biased coin lands up heads 80\% of the time and tails 20\% of the time, these values of heads and tails will occur with these percentages when this biased coin is tossed.' + \
        #             "Please only respond with a singular character, no extra spaces, characters, or quotes please: 'H' for heads and 'T' for tails. First I will provide two examples of this task, then you will perform the task yourself.\n " + \
        #             "Example #1: Let the probability that the biased coin lands heads up be 100\% and the probability that the coin lands tails up be 0\%. Flip this biased coin. \n Answer: H\n" + \
        #             "Example #2: Let the probability that the biased coin lands heads up be 20\% and the probability that the coin lands tails up be 80\%. Flip this biased coin.\n Answer: T\n" + \
        #             "Your turn! \n\n"
        prompt = read_file_to_string('{}/inputs/toy_problems/{}/prompt.txt'.format(os.getcwd(), question_type))

    # elif question_type=='biased_coin_seq':
    #     prompt = read_file_to_string('{}/inputs/toy_problems/biased_coin_seq/prompt.txt')
    # elif question_type=='biased_coin_seq_flipped':
    #     prompt = read_file_to_string('{}/inputs/toy_problems/{}/prompt_flipped.txt')


    elif question_type=='biased_dice':
        prompt = "I will specify a distribution for a biased dice. Please simulate a dice toss just as the probabilities for this biased dice are specified. As an example, if a dice lands on '1' 50\% of the time, '2' 0\% of the time,  '3' 0\% of the time, '4' 25\% of the time, '5' 0\% of the timne, and '6' 25\% of the time, these numbers will occur with these percentages when this biased dice is tossed." + \
            "Please only respond with a singular number, no extra spaces, characters, or quotes please: '1' if the dice lands on 1, '2' if the dice lands on 2, etc. First I will provide two examples of this task, then you will perform the task yourself.\n " + \
            "Example #1: Let the probability that the biased dice lands on '1' 20\% of the time, '2' 30\% of the time, '3' 50\% of the time, '4' 0\% of the time, '5' 0\% of the timne, and '6' 0\% of the time. Toss this biased dice. \n Answer: 2\n" + \
            "Example #2: Let the probability that the biased dice lands on '1' 5\% of the time, '2' 5\% of the time, '3' 5\% of the time, '4' 5\% of the time, '5' 20\% of the timne, and '6' 60\% of the time. Toss this biased dice. \n Answer: 6\n" + \
            "Your turn! \n\n"
    elif question_type=='biased_deck':
        prompt = "I will specify a distribution for a biased deck. Please simulate pulling a card from this biased deck just as the probabilities for this biased deck are specified. As an example, if the probabilities of the deck are drawing a '1' be 80\% likely, a '2' is 20\% likely, a '3' is 0\% likely, a '4' is 0\% likely, these cards will be drawn with these percentages when a card is pulled from this biased deck." + \
        "Please only respond with a singular character, no extra spaces, characters, or quotes please: return 'A' if the card pulled is 'A', 'B' if the card pulled is 'B', etc. First I will provide two examples of this task, then you will perform the task yourself.\n " + \
        "Example #1: A biased deck with 4 types of cards: A, B, C, and D. Let the probability of drawing a 'A' be 50%, drawing a 'B' be 20%, drawing a 'C' be 30%, drawing a 'D' be 0%. Pull a card from this biased deck. \n Answer: B\n" + \
        "Example #2: A biased deck with 4 types of cards: A, B, C, and D. Let the probability of drawing a 'A' be 0%, drawing a 'B' be 20%, drawing a 'C' be 80%, drawing a 'D' be 0%. Pull a card from this biased deck. \n Answer: C\n" + \
        "Your turn! \n\n"

    else: 
        prompt="After surveying the US population on the following question, option A is selected 30% of the time, option B is selected 50\% of the time, and option C is selected 20\% of the time. "+\
        "Please respond to the following question just as the survey respondent would respond to this question by following the distribution. More specifically, this means respond with option A 30% of the time, option B 50% of the time and option C' 20% of the time.\n"+\
        "\n Format your answer as a singular character 'A' for choice A, 'B' for choice B, 'C' for choice C. First I will provide two example of this task, then you will perform the task yourself.\n " + \
        "Example #1: \n What is your favorite snack? A. Pizza \n B. Chips\n C. Popcorn \n. \n Answer: B\n" + \
        "Example #2: \n What is your favorite book? A. 'The Night Circus' \n B. 'Sapiens A Brief History of Humankind' \n C. 'The Alchemist' \n\n Answer: C\n" + \
        "Your turn! "
    
    return prompt

# OPINION QA STUFF
def get_similar_qIDS(data_path, wave):
    similar_qIDS = []
    f = open(data_path + wave + '/question_similarity.json')
    data = json.load(f)
    # for each question in the wave that has more than 1 similar question ID
    for question_ID in data.keys():
        if len(data[question_ID]['similar_questionIDs'])>1:
            similar_qIDS.append(question_ID)
    print("Number of Question IDS: {}".format(len(similar_qIDS)))
    return similar_qIDS

# OPINION QA STUFF
def get_all_qIDS(data_path, wave):
    f = open(data_path + wave + '/question_similarity.json')
    data = json.load(f)
    return list(data.keys())

def get_q_IDs_waves_disagree(data_path, foldername, args=None):
    if args.dataset=='opinionqa':
        path = data_path+'/'+foldername+"/info.csv"
        df = pd.read_csv(path)
        waves = np.array(df['survey'])
        q_IDs = np.array(df['key'])
        return q_IDs, waves

    else: 
        path = data_path+'/'+foldername+"_data.json"
        f = open(path)
        data = json.load(f)
        q_IDs = list(data.keys())
        return q_IDs


def create_test_set(data_path, split=0.05):
    test_set_qIDS, test_set_waves = [], []
    # test set should contain ~10% of the wave's questions
    for wave in all_waves:
        f = open(data_path + wave + '/POLPARTY_data.json')
        data = json.load(f)
        n = int(len(data) * split)
        if n==0: n=1
        print(n)
        random.seed(10)
        sampled_qIDs = random.sample(data.keys(), n)
        test_set_qIDS += sampled_qIDs
        test_set_waves += [wave]*n 

    json_data = json.dumps({'test_set_qIDS': test_set_qIDS, 'test_set_waves': test_set_waves})
    with open(data_path+'/test_set.json', 'w') as json_file:
        json_file.write(json_data)
    return 

def get_test_set(data_path):
    # Open the JSON file
    with open(data_path+'/test_set.json', 'r') as json_file:
        # Load JSON data
        data = json.load(json_file)

    # Retrieve the lists from the loaded data
    test_set_qIDS = data['test_set_qIDS']
    test_set_waves = data['test_set_waves']
    return test_set_qIDS, test_set_waves


def get_taskalign_prompt(args, data, q_ID, demographic, wave, demographic_group, k=1):
    prompt=''
    if args.output_type=='sequence':
        # select a certain percentage 

        prompt+="I will specify a distribution of responses from {}s to questions on different topics. For the examples I provide, I will specify the distribution of responses (Context), ask the question (Question), and then provide the a sequence of 30 samples from this group of participants (Answer). Then you will perform the task yourself on a new question, where only the question is specified. You will need to rely on your prior knowledege from the previous examples to simulate 30 samples from this group of {}s on a new question. Please replicate the distribution of responses based on the prior examples. Please only respond with only 30 characters, no extra spaces, characters, quotes or text. Answers with more than 30 characters will not be accepted.".format(demographic, demographic)
        prompt+="\n\nGiven the fields `question`, produce the fields `answer`.\n---\n\nFollow the following format.\n\nQuestion: ${question}\nAnswer: ${answer} where you provide only the answer\n\n------\n\n"
                
        # INCONTEXT EX ARE k% of the training dataset 
        # go through all the waves and grab k% to put in the prompt
        for wave in all_waves:
            data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
            training_data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
            n_samples = int(len(data) * k * 0.01)
            training_data_potential_keys = list(training_data.keys())
            test_q_IDS, _ = get_test_set(data_path)
            # Remove overlapping keys from test and train
            training_data_potential_keys = [key for key in training_data_potential_keys if key not in test_q_IDS]
            if n_samples>len(training_data_potential_keys):
                n_samples=len(training_data_potential_keys)
            sampled_qIDs = random.sample(training_data_potential_keys, n_samples)
            # print(n_samples)
            for sample_qID in sampled_qIDs:
                all_options, probs = [], []
                prompt += "\nContext: Let the probability that a {} responds to the following question with ".format(demographic)
                n = (sum(training_data[sample_qID][demographic].values()))
                for i, option in enumerate(list(training_data[sample_qID][demographic].keys())):
                    all_options.append(options[i])
                    probs.append(training_data[sample_qID][demographic][option]/n)
                    prompt +="'{}' be {}%, ".format(options[i], int((training_data[sample_qID][demographic][option]/n)*100))
                prompt+= "Question: " + training_data[sample_qID]['question_text'] + "?\n"
                for i, option in enumerate(list(training_data[sample_qID][demographic].keys())):
                    prompt +="'{}'. {}\n".format(options[i], option)
                # Generate 30 flips
                flips = random.choices(all_options, probs, k=30)
                prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
        # prompt += "\n Context: Let the probability that a {} responds to the following question with ".format(demographic)
        # n = (sum(data[q_ID][demographic].values()))
        # for i, option in enumerate(list(data[q_ID][demographic].keys())):
        #     prompt +="'{}' be {}%, ".format(options[i], int((data[q_ID][demographic][option]/n)*100))
        prompt+= "Question: " + data[q_ID]['question_text'] + "?\n"
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="'{}'. {}\n".format(options[i], option)
        prompt+="Answer: "
    return prompt




def get_task4_prompt(args, data, q_ID, demographic, wave, demographic_group, k=1):
    prompt=''
    if args.output_type=='sequence':
        prompt+="I will specify a distribution of responses from {}s to 10 questions on different topics. For the examples I provide, I will specify the distribution of responses (Context), ask the question (Question), and then provide the a sequence of 30 samples from this group of participants (Answer). Then you will perform the task yourself on a new question, where only the question is specified. You will need to rely on your prior knowledege from the previous examples to simulate 30 samples from this group of {}s on a new question. Please replicate the distribution of responses based on the prior examples. Please only respond with only 30 characters, no extra spaces, characters, quotes or text. Answers with more than 30 characters will not be accepted.".format(demographic, demographic)
        prompt+="\n\nGiven the fields`question`, produce the fields `answer`.\n---\n\nFollow the following format.\n\nQuestion: ${question}\nAnswer: ${answer}\n\n------\n\n"
        
        # create the txt file out of similar questions 

        # INCONTEXT EX ARE SIMILAR Qs
        data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
        f = open(data_path + wave + '/question_dissimilarity.json')
        q_dissim_data = json.load(f)
        dissimilar_qIDS, dissimilar_qIDS_waves = q_dissim_data[q_ID]['dissimilar_questionIDs_{}'.format(str(k))], q_dissim_data[q_ID]['dissimilar_questionIDs_{}_waves'.format(str(k))]
        dissimilar_qIDS = dissimilar_qIDS[0]
        dissimilar_qIDS_waves = dissimilar_qIDS_waves[0]
        for i, dissimilar_qID in enumerate(dissimilar_qIDS):
            dissim_data = json.load(open(data_path + dissimilar_qIDS_waves[i] + '/' + demographic_group + "_data.json"))
            all_options, probs = [], []
            prompt += "\nContext: Let the probability that a {} responds to the following question with ".format(demographic)
            n = (sum(dissim_data[dissimilar_qID][demographic].values()))
            for i, option in enumerate(list(dissim_data[dissimilar_qID][demographic].keys())):
                all_options.append(options[i])
                probs.append(dissim_data[dissimilar_qID][demographic][option]/n)
                prompt +="'{}' be {}%, ".format(options[i], int((dissim_data[dissimilar_qID][demographic][option]/n)*100))
            prompt+= "Question: " + dissim_data[dissimilar_qID]['question_text'] + "?\n"
            for i, option in enumerate(list(dissim_data[dissimilar_qID][demographic].keys())):
                prompt +="'{}'. {}\n".format(options[i], option)
            # Generate 30 flips
            flips = random.choices(all_options, probs, k=30)
            prompt+="Answer: " + ' '.join(flips) + '\n\n------ '

        # prompt += "\n Context: Let the probability that a {} responds to the following question with ".format(demographic)
        # n = (sum(data[q_ID][demographic].values()))
        # for i, option in enumerate(list(data[q_ID][demographic].keys())):
        #     prompt +="'{}' be {}%, ".format(options[i], int((data[q_ID][demographic][option]/n)*100))
        prompt+= "Question: " + data[q_ID]['question_text'] + "?\n"
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="'{}'. {}\n".format(options[i], option)
        prompt+="Answer: "
            
    elif args.output_type=='model_logprobs':
        prompt+="I will specify a distribution of responses from {}s to 10 questions on different topics. For the examples I provide, I will specify the distribution of responses (Context), ask the question (Question), and then provide a sample response from this group of participants based on the specified distribution (Answer). Then you will perform the task yourself on a new question, where only the question is specified. You will need to rely on your prior knowledege from the previous examples to simulate one draw from this group of {}s on a new question. Please replicate the distribution of responses based on the prior examples. Please only respond with a singular character, corresponding to the multiple choice answer. Please no extra spaces, characters, quotes or text. Answers with more than one characters will not be accepted.".format(demographic, demographic)
        prompt+="\n\nGiven the fields`question`, produce the fields `answer`.\n---\n\nFollow the following format.\n\nQuestion: ${question}\nAnswer: ${answer}\n\n------\n\n"
        
        # create the txt file out of similar questions 

        # INCONTEXT EX ARE disSIMILAR Qs
        data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
        f = open(data_path + wave + '/question_dissimilarity.json')
        q_dissim_data = json.load(f)
        dissimilar_qIDS, dissimilar_qIDS_waves = q_dissim_data[q_ID]['dissimilar_questionIDs_{}'.format(str(k))], q_dissim_data[q_ID]['dissimilar_questionIDs_{}_waves'.format(str(k))]
        dissimilar_qIDS = dissimilar_qIDS[0]
        dissimilar_qIDS_waves = dissimilar_qIDS_waves[0]
        for i, dissimilar_qID in enumerate(dissimilar_qIDS):
            dissim_data = json.load(open(data_path + dissimilar_qIDS_waves[i] + '/' + demographic_group + "_data.json"))
            all_options, probs = [], []
            prompt += "\nContext: Let the probability that a {} responds to the following question with ".format(demographic)
            n = (sum(dissim_data[dissimilar_qID][demographic].values()))
            for i, option in enumerate(list(dissim_data[dissimilar_qID][demographic].keys())):
                all_options.append(options[i])
                probs.append(dissim_data[dissimilar_qID][demographic][option]/n)
                prompt +="'{}' be {}%, ".format(options[i], int((dissim_data[dissimilar_qID][demographic][option]/n)*100))
            prompt+= "Question: " + dissim_data[dissimilar_qID]['question_text'] + "?\n"
            for i, option in enumerate(list(dissim_data[dissimilar_qID][demographic].keys())):
                prompt +="'{}'. {}\n".format(options[i], option)
            # Generate 30 flips
            flips = random.choices(all_options, probs, k=1)
            prompt+="Answer: " + ' '.join(flips) + '\n\n------ '

        # prompt += "\n Context: Let the probability that a {} responds to the following question with ".format(demographic)
        # n = (sum(data[q_ID][demographic].values()))
        # for i, option in enumerate(list(data[q_ID][demographic].keys())):
        #     prompt +="'{}' be {}%, ".format(options[i], int((data[q_ID][demographic][option]/n)*100))
        prompt+= "Question: " + data[q_ID]['question_text'] + "?\n"
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="'{}'. {}\n".format(options[i], option)
        prompt+="Answer: "
    
    
    return prompt

def avg_incontext_ex(wave, q_ID, demographic, data):
    # INCONTEXT EX ARE SIMILAR Qs
    proportions=defaultdict(list)
    avg_proportions = {}
    data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
    f = open(data_path + wave + '/question_similarity.json')
    sim_data = json.load(f)
    similar_qIDS = sim_data[q_ID]['similar_questionIDs']
    for similar_qID in similar_qIDS:
        if similar_qID != q_ID:
            # only add this question if the number of MC choices in the similar questions matches the number of MC options in the qID
            # breakpoint()
            if len(data[q_ID][demographic].keys()) == len(data[similar_qID][demographic].keys()):
                n = (sum(data[similar_qID][demographic].values()))
                for i, option in enumerate(list(data[similar_qID][demographic].keys())):
                    proportions[options[i]].append(data[similar_qID][demographic][option]/n)

    # compute avg
    for key in proportions.keys():
        if proportions[key]: 
            avg_proportions[key] = np.mean(proportions[key])
    return avg_proportions


def get_ICL_qIDs(icl_data, q_ID, task3_type, wave, demographic, data_path, dataset='opinionqa'):
    if task3_type == 'all_but_one':
        ICL_qIDS = []
        # get wave from q_ID
        # read the info.csv file to map qID to wave
        df = pd.read_csv('{}/{}/info.csv'.format(data_path, wave))
        wave = df.loc[df['key'] == q_ID, 'survey']
        wave = wave.iloc[0][4:]
        # read in the wave data as icl_data
        wave_path = data_path+'/'+wave+'/'+'NONE_data.json'
        with open(wave_path, 'r') as file: icl_data = json.load(file)

        for qID_in_ICLdata in icl_data.keys():
            if qID_in_ICLdata != q_ID: ICL_qIDS.append(qID_in_ICLdata)
        return ICL_qIDS
        

    
    # easy (n=5): top 10 textually similar questions and use the 5 question that has the most ground truth output distribution similarity to the new question
    # Verify this is not too easy 
    elif task3_type == 'easy' or task3_type=='easy_hard': 
        # open up top 10 similar questions 
        if dataset=='opinionqa':
            data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
            f = open(data_path + wave + '/question_similarity_top10.json')

            question_similarity_top10 = json.load(f)
            top10 = question_similarity_top10[q_ID]

            # score these ICL qIDs based on similarity in distribution to q_ID
            # first separate the qIDs with the equivalent num MC and not equivalent num MC 
            easy, easy_hard = [], []
            n_qID_MC = len(icl_data[q_ID]['MC_options'])
            for icl_qID in top10: 
                if n_qID_MC == len(icl_data[icl_qID]['MC_options']): easy.append(icl_qID)
                else: easy_hard.append(icl_qID)
            
            if len(easy_hard)==5 and task3_type=='easy_hard': return easy_hard
            if len(easy)==5 and task3_type=='easy': return easy

            # if len(easy)<len(easy_hard): There are too many that have different MC options
            # Since we want to keep the more challenging ones in easy_hard, move the textually similar ones to easy 
            # the IDs in easy_hard are ranked from most textually similar to least 
            if len(easy)<len(easy_hard):
                num_to_transfer = len(easy_hard) - 5
                easy = easy + easy_hard[:num_to_transfer]
                easy_hard = easy_hard[num_to_transfer:]

            # if len(easy)>len(easy_hard): split based on similarity to ground truth distribution 
            if len(easy)>len(easy_hard):
                num_to_transfer = len(easy) - 5
                distrib_dist = []
                q_ID_values = np.array(list(icl_data[q_ID][demographic].values()))/np.sum(list(icl_data[q_ID][demographic].values()))
                for easy_qID in easy: 
                    icl_values = []
                    # calculate distributional difference
                    for MC_option in icl_data[easy_qID]['MC_options']:
                        if MC_option in icl_data[easy_qID][demographic].keys(): 
                            icl_values.append(icl_data[easy_qID][demographic][MC_option])
                        else: icl_values.append(0)
                        
                    icl_values = np.array(icl_values)/np.sum(icl_values)
                    try: distrib_dist.append(total_variation(icl_values, q_ID_values))
                    except: distrib_dist.append(1)
                sorted_pairs = sorted(zip(distrib_dist, easy))
                # order qIDs based on smallest to largest distribution difference
                sorted_list_qIDs= [pair[1] for pair in sorted_pairs]

                # keep the ones in easy that have the lowest distributional differences 
                easy = sorted_list_qIDs[:5]
                easy_hard = easy_hard + sorted_list_qIDs[5:]

        elif dataset=='global_values': 

            data_path = '{}/globalvalues/question_similarity.json'.format(os.getcwd())
            f = open(data_path)

            question_similarity_top10 = json.load(f)
            top5 = list(question_similarity_top10[q_ID][demographic].keys())[:5]
            easy_hard = top5
            easy = top5


        elif dataset=='nytimes': 
            f = open(data_path + '/question_similarity_top10.json')
            question_similarity_top10 = json.load(f)
            top10 = question_similarity_top10[q_ID]

            # score these ICL qIDs based on similarity in distribution to q_ID
            # first separate the qIDs with the equivalent num MC and not equivalent num MC 
            easy, easy_hard = [], []
            n_qID_MC = len(icl_data[q_ID]['MC_options'])
            for icl_qID in top10: 
                if n_qID_MC == len(icl_data[icl_qID]['MC_options']): easy.append(icl_qID)
                else: easy_hard.append(icl_qID)
            
            if len(easy_hard)==5 and task3_type=='easy_hard': return easy_hard
            if len(easy)==5 and task3_type=='easy': return easy

            # if len(easy)<len(easy_hard): There are too many that have different MC options
            # Since we want to keep the more challenging ones in easy_hard, move the textually similar ones to easy 
            # the IDs in easy_hard are ranked from most textually similar to least 
            if len(easy)<len(easy_hard):
                num_to_transfer = len(easy_hard) - 5
                easy = easy + easy_hard[:num_to_transfer]
                easy_hard = easy_hard[num_to_transfer:]

            # if len(easy)>len(easy_hard): split based on similarity to ground truth distribution 
            if len(easy)>len(easy_hard):
                num_to_transfer = len(easy) - 5
                q_ID_values = np.array(list(icl_data[q_ID][demographic].values()))/np.sum(list(icl_data[q_ID][demographic].values()))
                for easy_qID in easy: 
                    icl_values = []
                    # calculate distributional difference
                    for MC_option in icl_data[easy_qID]['MC_options']:
                        if MC_option in icl_data[easy_qID][demographic].keys(): 
                            icl_values.append(icl_data[easy_qID][demographic][MC_option])
                        else: icl_values.append(0)
                        
                    icl_values = np.array(icl_values)/np.sum(icl_values)

                sorted_pairs = sorted(zip(distrib_dist, easy))
                # order qIDs based on smallest to largest distribution difference
                sorted_list_qIDs= [pair[1] for pair in sorted_pairs]

                # keep the ones in easy that have the lowest distributional differences 
                easy = sorted_list_qIDs[:5]
                easy_hard = easy_hard + sorted_list_qIDs[5:]

        assert(len(easy_hard)==len(easy))
        if len(easy_hard)==5 and task3_type=='easy_hard': return easy_hard
        if len(easy)==5 and task3_type=='easy': return easy

        # if the q_ID 

    # easy hard (n=5): top 10 textually similar questions and use the 5 question that has the least ground truth output distribution similarity to the new question

    else: breakpoint()

    return 

def get_prompt_nytimes(args, model, question_type, data, q_ID, demographic, wave, demographic_group, k=None, ficticious_group_ablation=False, shuffled_incontext_labels=False):
    prompt_names = {"Republican" : 'a Republican person', "Democrat" : 'a Democrat person', "Male" : 'a man', "Female" : 'a woman'}
    if demographic in prompt_names.keys(): demographic_in_prompt = prompt_names[demographic]
    demographic_in_prompt = demographic
    prompt = ''

    if question_type=='task0':
        if args.output_type=='sequence':
            prompt = 'I will provide a sequence of responses to a series of questions to help you understand the formatting of this task. After the 5 examples, please simulate 30 samples for the question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself.'
        elif args.output_type=='model_logprobs': 
            prompt = 'I will provide a multiple choice response to a series of questions to help you understand the formatting of this task. After the 5 examples, please simulate an answer for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself.'
        elif args.output_type=='express_distribution': 
            prompt = 'I will provide a distribution over answer choices on a series of questions to help you understand the formatting of this task. After the 5 examples, please express the distribution of answers for the question asked. Do not follow the distribution expressed in the examples as they are only to give you an example of formatting, express the distribution based on your understanding of how humans may respond. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself.'


        prompt+= read_file_to_string('{}/inputs/nytimes/{}/prompt_{}.txt'.format(os.getcwd(), args.output_type, question_type))
        prompt+='\nYour turn!\n'
        if model == 'llama3-70b' and args.output_type=='model_logprobs': prompt= 'Please simulate an answer for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. '

        prompt+= "\nBook Title: " + q_ID 
        prompt+= "\nBook Genre: " + data[q_ID]['genre'] 
        prompt+= "\nBook Summary: " + data[q_ID]['summary'] 
        prompt+='\nQuestion: Given the information about this book, how likely are you to read it?\n'
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="{}. {}\n".format(options[i], option)
        prompt+="Answer:"
        
    elif question_type=='task1':
        if args.output_type=='sequence':
            prompt = 'I will provide a sequence of responses to a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please simulate 30 samples from a group of "{}" for the question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)
        elif args.output_type=='model_logprobs': 
            prompt = 'I will provide a multiple choice response to a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)
        elif args.output_type=='express_distribution': 
            prompt = 'I will provide a distribution over answer choices on a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please express the distribution of answers from a group of "{}" for the question asked. Do not follow the distribution expressed in the examples as they are only to give you an example of formatting, express the distribution based on your understanding of how humans may respond. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)

        prompt+= read_file_to_string('{}/inputs/nytimes/{}/prompt_{}.txt'.format(os.getcwd(), args.output_type, question_type))
        prompt+='\nYour turn!\n'
        if model == 'llama3-70b' and args.output_type=='model_logprobs': prompt= 'Please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)

        prompt+= "\nBook Title: " + q_ID 
        prompt+= "\nBook Genre: " + data[q_ID]['genre'] 
        prompt+= "\nBook Summary: " + data[q_ID]['summary']
        prompt+='\nQuestion: Given the information about this book, how likely is a {} to read it?\n'.format(demographic_in_prompt)
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="{}. {}\n".format(options[i], option)
        prompt+="Answer:"

    elif question_type=='task3':

        prompt = "In this task you will receive information on the distribution of responses from a group of {}s to related questions. Given this data, your task is to simulate an answer to a new question from the group of {}s. ".format(demographic_in_prompt, demographic_in_prompt)
        prompt+= "First, I will provide the distribution of responses from a group of {}s to a series of questions. Afterwards, I will provide example responses to the question to help you understand the formatting of this task. ".format(demographic_in_prompt)

        if args.output_type=='sequence':
            prompt+= 'After the examples, please simulate 30 samples from a group of {} for the new question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)
        elif args.output_type=='model_logprobs': 
            prompt += 'After the examples, please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)
        elif args.output_type=='express_distribution': 
            prompt += 'After the examples, please express the distribution of answers from a group of "{}" for the question asked. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)

        prompt+="\n\nGiven the fields `context`, `Book Title`, `Book Genre`, `Book Summary`, `question`, produce the fields `answer`. Your task will not have `context`.\n\n------\n\n"


        # ICL Is all other question in the wave 
        
        icl_data = data
        data_path = '{}/nytimes/'.format(os.getcwd())
        ICL_qIDS = get_ICL_qIDs(icl_data, q_ID, args.task3_type, wave, demographic, data_path, dataset='nytimes')
        for icl_qID in ICL_qIDS:
            if icl_qID != q_ID:
                
                all_options, probs = [], []
                prompt += "\nContext: Let the probability that a {} responds to the following question with ".format(demographic_in_prompt)

                MC_options = data[icl_qID]['MC_options']
                n = (sum(data[icl_qID][demographic].values()))
                
                for i, option in enumerate(MC_options):
                    if str(option) in data[icl_qID][demographic]: 
                        all_options.append(options[i])
                        probs.append(data[icl_qID][demographic][option]/n)
                        prompt +="'{}' be {}%, ".format(option, int((data[icl_qID][demographic][option]/n)*100))
                prompt+= ".\nBook Title: " + icl_qID 
                prompt+= "\nBook Genre: " + data[icl_qID]['genre'] 
                prompt+= "\nBook Summary: " + data[icl_qID]['summary'] 
                prompt+='\nQuestion: Given the information about this book, how likely is a {} to read it?\n'.format(demographic_in_prompt)
                for i, option in enumerate(MC_options):
                    prompt +="'{}'. {}\n".format(options[i], option)
                if args.output_type=='sequence':
                    # Generate 30 flips
                    try: 
                        flips = random.choices(all_options, probs, k=30)
                        prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                    except: prompt+=''
                    
                elif args.output_type=='model_logprobs': 
                    try: 
                        flips = random.choices(all_options, probs, k=1)
                        prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                    except: prompt+=''
                elif args.output_type=='express_distribution': 
                    prompt +="Answer: {"
                    for i, prob in enumerate(probs):
                        prompt+="{}: '{}%', ".format(all_options[i], int(prob*100))
                    prompt = prompt[:-2] + '}\n\n------ ' # -2 to get rid of last space

        prompt+='\nYour turn! Please answer this question for the group of {}s. As a reminder, this group is the exact same group as the group in the previous examples. The previous examples are used to provide an example of formatting and to give you insight into how this group would respond to such questions.\n'.format(demographic_in_prompt)


        if model == 'llama3-70b' and args.output_type=='model_logprobs': 
            prompt=''
            for icl_qID in ICL_qIDS:
                if icl_qID != q_ID:
                    
                    all_options, probs = [], []
                    prompt += "Let the probability that a {} responds to the following question with ".format(demographic_in_prompt)

                    MC_options = data[icl_qID]['MC_options']
                    n = (sum(data[icl_qID][demographic].values()))
                    
                    for i, option in enumerate(MC_options):
                        if str(option) in data[icl_qID][demographic]: 
                            all_options.append(options[i])
                            probs.append(data[icl_qID][demographic][option]/n)
                            prompt +="'{}' be {}%, ".format(option, int((data[icl_qID][demographic][option]/n)*100))
                    prompt+= ".\nBook Title: " + icl_qID 
                    prompt+= "\nBook Genre: " + data[icl_qID]['genre'] 
                    prompt+= "\nBook Summary: " + data[icl_qID]['summary'] 
                    prompt+='\nQuestion: Given the information about this book, how likely is a {} to read it?\n'.format(demographic_in_prompt)
                    for i, option in enumerate(MC_options):
                        prompt +="'{}'. {}\n".format(options[i], option)
                    
                    try: 
                        flips = random.choices(all_options, probs, k=1)
                        prompt+="Answer: " + ' '.join(flips) + '\n'
                    except: prompt+=''
                    

            prompt+='\nPlease answer this question for the group of {}s. \n'.format(demographic_in_prompt)

        prompt+= "\nBook Title: " + q_ID 
        prompt+= "\nBook Genre: " + data[q_ID]['genre'] 
        prompt+= "\nBook Summary: " + data[q_ID]['summary'] 
        prompt+='\nQuestion: Given the information about this book, how likely is a {} to read it?\n'.format(demographic_in_prompt)

        
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="'{}'. {}\n".format(options[i], option)
        prompt+="Answer:"
    
    elif question_type=='task5':

        prompt = "In this task you will receive information on the distribution of responses from a group of people to related questions. Given this data, your task is to simulate an answer to a new question from the same group of people. "
        prompt+= "First, I will provide the distribution of responses from a group of people to a series of questions. Afterwards, I will provide example responses to the question to help you understand the formatting of this task. "

        if args.output_type=='sequence':
            prompt+= 'After the examples, please simulate 30 samples from the group of people for the new question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'
        elif args.output_type=='model_logprobs': 
            prompt += 'After the examples, please simulate an answer from the group of people for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'
        elif args.output_type=='express_distribution': 
            prompt += 'After the examples, please express the distribution of answers from the group of people for the question asked. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'

        prompt+="\n\nGiven the fields `context`, `Book Title`, `Book Genre`, `Book Summary`, `question`, produce the fields `answer`. Your task will not have `context`.\n\n------\n\n"


        # ICL Is all other question in the wave 
        
        icl_data = data
        data_path = '{}/nytimes/'.format(os.getcwd())
        ICL_qIDS = get_ICL_qIDs(icl_data, q_ID, args.task3_type, wave, demographic, data_path, dataset='nytimes')
        for icl_qID in ICL_qIDS:
            if icl_qID != q_ID:
                
                all_options, probs = [], []
                prompt += "\nContext: Let the probability that people from this group respond to the following question with "

                MC_options = data[icl_qID]['MC_options']
                n = (sum(data[icl_qID][demographic].values()))
                
                for i, option in enumerate(MC_options):
                    if str(option) in data[icl_qID][demographic]: 
                        all_options.append(options[i])
                        probs.append(data[icl_qID][demographic][option]/n)
                        prompt +="'{}' be {}%, ".format(option, int((data[icl_qID][demographic][option]/n)*100))
                prompt+= ".\nBook Title: " + icl_qID 
                prompt+= "\nBook Genre: " + data[icl_qID]['genre'] 
                prompt+= "\nBook Summary: " + data[icl_qID]['summary'] 
                prompt+='\nQuestion: Given the information about this book, how likely is a a person from this group to read it?\n'
                for i, option in enumerate(MC_options):
                    prompt +="'{}'. {}\n".format(options[i], option)
                if args.output_type=='sequence':
                    # Generate 30 flips
                    try: 
                        flips = random.choices(all_options, probs, k=30)
                        prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                    except: prompt+=''
                    
                elif args.output_type=='model_logprobs': 
                    try: 
                        flips = random.choices(all_options, probs, k=1)
                        prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                    except: prompt+=''
                elif args.output_type=='express_distribution': 
                    prompt +="Answer: {"
                    for i, prob in enumerate(probs):
                        prompt+="{}: '{}%', ".format(all_options[i], int(prob*100))
                    prompt = prompt[:-2] + '}\n\n------ ' # -2 to get rid of last space

        prompt+='\nYour turn! Please answer this question for the group of people. As a reminder, this group is the exact same group as the group in the previous examples. The previous examples are used to provide an example of formatting and to give you insight into how this group would respond to such questions.\n'


        if model == 'llama3-70b' and args.output_type=='model_logprobs': 
            prompt=''
            for icl_qID in ICL_qIDS:
                if icl_qID != q_ID:
                    
                    all_options, probs = [], []
                    prompt += "Let the probability that a {} responds to the following question with ".format(demographic_in_prompt)

                    MC_options = data[icl_qID]['MC_options']
                    n = (sum(data[icl_qID][demographic].values()))
                    
                    for i, option in enumerate(MC_options):
                        if str(option) in data[icl_qID][demographic]: 
                            all_options.append(options[i])
                            probs.append(data[icl_qID][demographic][option]/n)
                            prompt +="'{}' be {}%, ".format(option, int((data[icl_qID][demographic][option]/n)*100))
                    prompt+= ".\nBook Title: " + icl_qID 
                    prompt+= "\nBook Genre: " + data[icl_qID]['genre'] 
                    prompt+= "\nBook Summary: " + data[icl_qID]['summary'] 
                    prompt+='\nQuestion: Given the information about this book, how likely is a {} to read it?\n'.format(demographic_in_prompt)
                    for i, option in enumerate(MC_options):
                        prompt +="'{}'. {}\n".format(options[i], option)
                    
                    try: 
                        flips = random.choices(all_options, probs, k=1)
                        prompt+="Answer: " + ' '.join(flips) + '\n'
                    except: prompt+=''
                    

            prompt+='\nPlease answer this question for the group of {}s. \n'.format(demographic_in_prompt)

        prompt+= "\nBook Title: " + q_ID 
        prompt+= "\nBook Genre: " + data[q_ID]['genre'] 
        prompt+= "\nBook Summary: " + data[q_ID]['summary'] 
        prompt+='\nQuestion: Given the information about this book, how likely is a person from this group to read it?\n'

        
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="'{}'. {}\n".format(options[i], option)
        prompt+="Answer:"
    # breakpoint()
    return prompt



def get_prompt_globalvalues(args, question_type, data, q_ID, demographic, wave, demographic_group, k=None, ficticious_group_ablation=False, shuffled_incontext_labels=False):
    demographic_in_prompt = demographic
    prompt = ''

    if question_type=='task0':
        if args.output_type=='sequence':
            prompt = 'I will provide a sequence of responses to a series of questions to help you understand the formatting of this task. After the 5 examples, please simulate 30 samples for the question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself.'
        elif args.output_type=='model_logprobs': 
            prompt = 'I will provide a multiple choice response to a series of questions to help you understand the formatting of this task. After the 5 examples, please simulate an answer for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself.'
        elif args.output_type=='express_distribution': 
            prompt = 'I will provide a distribution over answer choices on a series of questions to help you understand the formatting of this task. After the 5 examples, please express the distribution of answers for the question asked. Do not follow the distribution expressed in the examples as they are only to give you an example of formatting, express the distribution based on your understanding of how humans may respond. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself.'

        prompt+= read_file_to_string('{}/inputs/globalvalues/{}/prompt_{}.txt'.format(os.getcwd(), args.output_type, question_type))
        
        prompt+= "\nQuestion: " + q_ID + "?\n"
        for i, option in enumerate(data[q_ID]['options']):
            prompt +="'{}'. {}\n".format(options[i], option)
        prompt+="Answer: "
        

    elif question_type=='task1':
        
        if args.output_type=='sequence':
            prompt = 'I will provide a sequence of responses to a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please simulate 30 samples from a group of "{}" for the question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)
        elif args.output_type=='model_logprobs': 
            prompt = 'I will provide a multiple choice response to a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)
        elif args.output_type=='express_distribution': 
            prompt = 'I will provide a distribution over answer choices on a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please express the distribution of answers from a group of "{}" for the question asked. Do not follow the distribution expressed in the examples as they are only to give you an example of formatting, express the distribution based on your understanding of how humans may respond. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)

        prompt+= read_file_to_string('{}/inputs/globalvalues/{}/prompt_{}.txt'.format(os.getcwd(), args.output_type, question_type))
        prompt+='\nYour turn!\n'
        prompt+='\nHow would someone from {} respond to the following question?'.format(demographic_in_prompt)
        prompt+= "\nQuestion: " + q_ID + "?\n"
        for i, option in enumerate(data[q_ID]['options']):
            prompt +="'{}'. {}\n".format(options[i], option)
        prompt+="Answer: "
        
        

    elif question_type=='task3':

        prompt = "In this task you will receive information on the distribution of responses from a group of {}s to related survey questions. Given this data, your task is to simulate an answer to a new question from the group of {}s. ".format(demographic_in_prompt, demographic_in_prompt)
        prompt+= "First, I will provide the distribution of responses from a group of {}s to a series of questions in a section titled 'Data'. Afterwards, I will provide 5 example responses to the question to help you understand the formatting of this task. ".format(demographic_in_prompt)

        if args.output_type=='sequence':
            prompt+= 'After the examples, please simulate 30 samples from a group of {} for the new question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)
        elif args.output_type=='model_logprobs': 
            prompt += 'After the examples, please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)
        elif args.output_type=='express_distribution': 
            prompt += 'After the examples, please express the distribution of answers from a group of "{}" for the question asked. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)

        prompt+="\n\nGiven the fields 'context` and `question`, produce the fields `answer`. Your task will not have `context`.\n\n------\n\n"
        
        # ICL Is all other question in the wave 
        data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
        if wave=='Pew_American_Trends_Panel_disagreement_100': wave='Pew_American_Trends_Panel_disagreement_500'
        f = open('{}/{}/{}_data.json'.format(data_path, wave, demographic_group))
        icl_data = json.load(f)

        ICL_qIDS = get_ICL_qIDs(icl_data, q_ID, args.task3_type, wave, demographic, data_path)
        qID_to_wave = json.load(open('{}/qID_to_wave.json'.format(data_path)))
        for icl_qID in ICL_qIDS:
            if icl_qID != q_ID:
                if args.task3_type != 'all_but_one': 
                    ICL_qID_wave = qID_to_wave[q_ID]
                else: 
                    df_temp = pd.read_csv('{}/{}/info.csv'.format(data_path, wave))
                    ICL_qID_wave = df_temp.loc[df_temp['key'] == q_ID, 'survey']
                    ICL_qID_wave = ICL_qID_wave.iloc[0][4:]
                f = open('{}/{}/{}_data.json'.format(data_path, ICL_qID_wave, demographic_group))
                q_ID_data = json.load(f) # this is specific to the new wave 
                all_options, probs = [], []
                prompt += "\nContext: Let the probability that a {} responds to the following question with ".format(demographic_in_prompt)
                n = (sum(q_ID_data[icl_qID][demographic].values()))
                MC_options = list(q_ID_data[icl_qID][demographic].keys())
                if shuffled_incontext_labels: 
                    random.seed(random.randint(1, 1000))
                    random.shuffle(MC_options)
                for i, option in enumerate(MC_options):
                    all_options.append(options[i])
                    probs.append(q_ID_data[icl_qID][demographic][option]/n)
                    prompt +="'{}' be {}%, ".format(option, int((q_ID_data[icl_qID][demographic][option]/n)*100))
                prompt+= "\nQuestion: " + q_ID_data[icl_qID]['question_text'] + "?\n"
                for i, option in enumerate(MC_options):
                    prompt +="'{}'. {}\n".format(options[i], option)
                if args.output_type=='sequence':
                    # Generate 30 flips
                    try: 
                        flips = random.choices(all_options, probs, k=30)
                        prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                    except: prompt+=''
                    
                elif args.output_type=='model_logprobs': 
                    try: 
                        flips = random.choices(all_options, probs, k=1)
                        prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                    except: prompt+=''
                elif args.output_type=='express_distribution': 
                    prompt +="Answer: {"
                    for i, prob in enumerate(probs):
                        prompt+="'{}': '{}%', ".format(all_options[i], int(prob*100))
                    prompt = prompt[:-2] + '}\n\n------ ' # -2 to get rid of last space

        prompt+='\nYour turn! Please answer this question for the group of {}s. As a reminder, this group is the exact same group as the group in the previous examples. The previous examples are used to provide an example of formatting and to give you insight into how this group would respond to such questions.\n'.format(demographic_in_prompt)

        prompt+= "Question: " + data[q_ID]['question_text'] + "?\n"
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="'{}'. {}\n".format(options[i], option)
        prompt+="Answer: "


    return prompt



def get_prompt_opinionqa(args, model, question_type, data, q_ID, demographic, wave, demographic_group, k=None, ficticious_group_ablation=False, shuffled_incontext_labels=False):
    if ficticious_group_ablation: demographic_in_prompt=ficticious_group_ablation_mapping[demographic]
    else: demographic_in_prompt = demographic
    prompt = ''

    if question_type=='task0':
        if args.output_type=='sequence':
            prompt = 'I will provide a sequence of responses to a series of questions to help you understand the formatting of this task. After the 5 examples, please simulate 30 samples for the question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. '
        elif args.output_type=='model_logprobs': 
            prompt = 'I will provide a multiple choice response to a series of questions to help you understand the formatting of this task. After the 5 examples, please simulate an answer for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. '
        elif args.output_type=='express_distribution': 
            prompt = 'I will provide a distribution over answer choices on a series of questions to help you understand the formatting of this task. After the 5 examples, please express the distribution of answers for the question asked. Do not follow the distribution expressed in the examples as they are only to give you an example of formatting, express the distribution based on your understanding of how humans may respond. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. '

        prompt+='First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself.'
        prompt+= read_file_to_string('{}/inputs/opinion_surveys/{}/prompt_{}.txt'.format(os.getcwd(), args.output_type, question_type))
        prompt+='\nYour turn! '
        if model == 'llama3-70b' and args.output_type=='model_logprobs': prompt = "Please simulate an answer for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. "
        prompt+= "\nQuestion: " + data[q_ID]['question_text'] + "?\n"
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="{}. {}\n".format(options[i], option)
        prompt+="Answer:"

    elif question_type=='task1':
        if args.output_type=='sequence':
            prompt = 'I will provide a sequence of responses to a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please simulate 30 samples from a group of "{}" for the question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)
        elif args.output_type=='model_logprobs': 
            prompt = 'I will provide a multiple choice response to a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)
        elif args.output_type=='express_distribution': 
            prompt = 'I will provide a distribution over answer choices on a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please express the distribution of answers from a group of "{}" for the question asked. Do not follow the distribution expressed in the examples as they are only to give you an example of formatting, express the distribution based on your understanding of how humans may respond. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)

        prompt+= read_file_to_string('{}/inputs/opinion_surveys/{}/prompt_{}.txt'.format(os.getcwd(), args.output_type, question_type))
        prompt+='\nYour turn! Please answer this question for the group "{}". As a reminder, this group is different than the group in the previous examples. The previous examples are just used to provide an example of formatting.\n'.format(demographic_in_prompt)
        if model == 'llama3-70b' and args.output_type=='model_logprobs': prompt = 'Please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)
        prompt+= "\nQuestion: " + data[q_ID]['question_text'] + "?\n"
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="{}. {}\n".format(options[i], option)
        prompt+="Answer:"

    elif question_type=='task2':
        if args.output_type=='sequence':
            prompt = 'I will provide a sequence of responses to a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please simulate 30 samples from a group of "{}" for the question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)
        elif args.output_type=='model_logprobs': 
            prompt = 'I will provide a multiple choice response to a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)
        elif args.output_type=='express_distribution': 
            prompt = 'I will provide a distribution over answer choices on a series of questions from a random group to help you understand the formatting of this task. After the 5 examples, please express the distribution of answers from a group of "{}" for the question asked. Do not follow the distribution expressed in the examples as they are only to give you an example of formatting, express the distribution based on your understanding of how humans may respond. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for "{}".'.format(demographic_in_prompt, demographic_in_prompt)
        
        # if shuffled_incontext_labels:
        #     prompt += read_file_to_string('{}/inputs/opinion_surveys/prompt_{}_shuffled.txt'.format(os.getcwd(), question_type))
        prompt+= read_file_to_string('{}/inputs/opinion_surveys/{}/prompt_{}.txt'.format(os.getcwd(), args.output_type, question_type))

        prompt+='\nYour turn! Please answer this question for the group "{}". As a reminder, this group is different than the group in the previous examples. The previous examples are just used to provide an example of formatting.\n'.format(demographic_in_prompt)

        prompt += "\nContext: Let the probability that a {} responds to the following question with ".format(demographic_in_prompt)
        n = (sum(data[q_ID][demographic].values()))
        MC_options = list(data[q_ID][demographic].keys())
        
        if shuffled_incontext_labels: 
            random.seed(random.randint(1, 1000))
            random.shuffle(MC_options)
       
        for i, option in enumerate(MC_options):
            prompt +="'{}' be {}%, ".format(option, int((data[q_ID][demographic][option]/n)*100))
            # prompt +="'{}.' be {}%, ".format(options[i] int((data[q_ID][demographic][option]/n)*100))
        
        prompt+= "\nQuestion: " + data[q_ID]['question_text'] + "?\n"
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="'{}'. {}\n".format(options[i], option)
        prompt+="Answer: "

    elif question_type=='task3':


        prompt = "In this task you will receive information on the distribution of responses from a group of {}s to related survey questions. Given this data, your task is to simulate an answer to a new question from the group of {}s. ".format(demographic_in_prompt, demographic_in_prompt)
        prompt+= "First, I will provide the distribution of responses from a group of {}s to a series of questions in a section titled 'Data'. Afterwards, I will provide 5 example responses to the question to help you understand the formatting of this task. ".format(demographic_in_prompt)

        if args.output_type=='sequence':
            prompt+= 'After the examples, please simulate 30 samples from a group of {} for the new question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)
        elif args.output_type=='model_logprobs': 
            prompt += 'After the examples, please simulate an answer from a group of "{}" for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)
        elif args.output_type=='express_distribution': 
            prompt += 'After the examples, please express the distribution of answers from a group of "{}" for the question asked. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'.format(demographic_in_prompt)

        prompt+="\n\nGiven the fields 'context` and `question`, produce the fields `answer`. Your task will not have `context`.\n\n------\n\n"
        
        # ICL Is all other question in the wave 
        data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
        if wave=='Pew_American_Trends_Panel_disagreement_100': wave='Pew_American_Trends_Panel_disagreement_500'
        f = open('{}/{}/{}_data.json'.format(data_path, wave, demographic_group))
        icl_data = json.load(f)

        ICL_qIDS = get_ICL_qIDs(icl_data, q_ID, args.task3_type, wave, demographic, data_path)
        qID_to_wave = json.load(open('{}/qID_to_wave.json'.format(data_path)))
        for icl_qID in ICL_qIDS:
            if icl_qID != q_ID:
                if args.task3_type != 'all_but_one': 
                    ICL_qID_wave = qID_to_wave[q_ID]
                else: 
                    df_temp = pd.read_csv('{}/{}/info.csv'.format(data_path, wave))
                    ICL_qID_wave = df_temp.loc[df_temp['key'] == q_ID, 'survey']
                    ICL_qID_wave = ICL_qID_wave.iloc[0][4:]
                f = open('{}/{}/{}_data.json'.format(data_path, ICL_qID_wave, demographic_group))
                q_ID_data = json.load(f) # this is specific to the new wave 
                all_options, probs = [], []
                prompt += "\nContext: Let the probability that a {} responds to the following question with ".format(demographic_in_prompt)
                n = (sum(q_ID_data[icl_qID][demographic].values()))
                MC_options = list(q_ID_data[icl_qID][demographic].keys())
                if shuffled_incontext_labels: 
                    random.seed(random.randint(1, 1000))
                    random.shuffle(MC_options)
                for i, option in enumerate(MC_options):
                    all_options.append(options[i])
                    probs.append(q_ID_data[icl_qID][demographic][option]/n)
                    prompt +="{} be {}%, ".format(option, int((q_ID_data[icl_qID][demographic][option]/n)*100))
                prompt+= "\nQuestion: " + q_ID_data[icl_qID]['question_text'] + "?\n"
                for i, option in enumerate(MC_options):
                    # prompt +="{}. {}\n".format(options[i], option)
                    prompt +="{}. {}. ".format(options[i], option)
                if args.output_type=='sequence':
                    # Generate 30 flips
                    try: 
                        flips = random.choices(all_options, probs, k=30)
                        prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                    except: prompt+=''
                    
                elif args.output_type=='model_logprobs': 
                    try: 
                        flips = random.choices(all_options, probs, k=1)
                        prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                    except: prompt+=''
                    
                elif args.output_type=='express_distribution': 
                    prompt +="Answer: {"
                    for i, prob in enumerate(probs):
                        prompt+="'{}': '{}%', ".format(all_options[i], int(prob*100))
                    prompt = prompt[:-2] + '}\n\n------ ' # -2 to get rid of last space

        prompt+='\nYour turn! Please answer this question for the group of {}s. As a reminder, this group is the exact same group as the group in the previous examples. The previous examples are used to provide an example of formatting and to give you insight into how this group would respond to such questions.\n'.format(demographic_in_prompt)

        prompt+= "Question: " + data[q_ID]['question_text'] + "?\n"
        if model == 'llama3-70b' and args.output_type=='model_logprobs': 
            prompt=''
            for icl_qID in ICL_qIDS:
                if icl_qID != q_ID:
                    if args.task3_type != 'all_but_one': 
                        ICL_qID_wave = qID_to_wave[q_ID]
                    else: 
                        df_temp = pd.read_csv('{}/{}/info.csv'.format(data_path, wave))
                        ICL_qID_wave = df_temp.loc[df_temp['key'] == q_ID, 'survey']
                        ICL_qID_wave = ICL_qID_wave.iloc[0][4:]
                    f = open('{}/{}/{}_data.json'.format(data_path, ICL_qID_wave, demographic_group))
                    q_ID_data = json.load(f) # this is specific to the new wave 
                    all_options, probs = [], []
                    prompt += "\nLet the probability that a {} responds to the following question with ".format(demographic_in_prompt)
                    n = (sum(q_ID_data[icl_qID][demographic].values()))
                    MC_options = list(q_ID_data[icl_qID][demographic].keys())
                    if shuffled_incontext_labels: 
                        random.seed(random.randint(1, 1000))
                        random.shuffle(MC_options)
                    for i, option in enumerate(MC_options):
                        all_options.append(options[i])
                        probs.append(q_ID_data[icl_qID][demographic][option]/n)
                        prompt +="{} be {}%, ".format(option, int((q_ID_data[icl_qID][demographic][option]/n)*100))
                    prompt+= ": " + q_ID_data[icl_qID]['question_text'] + "?\n"
                    for i, option in enumerate(MC_options):
                        # prompt +="{}. {}\n".format(options[i], option)
                        prompt +="{}. {}. ".format(options[i], option)
                    
                    try: 
                        flips = random.choices(all_options, probs, k=1)
                        prompt+="Answer: " + ' '.join(flips) + '\n'
                    except: prompt+=''
            prompt+='Please answer this question for the group of {}s: '.format(demographic_in_prompt)
            prompt+=data[q_ID]['question_text'] + "?\n"
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="{}. {}. ".format(options[i], option)
        prompt+="\nAnswer:"

    elif question_type=='task4':
        prompt = get_task4_prompt(args, data, q_ID, demographic, wave, demographic_group, k=k)
    elif question_type=='align':
        prompt = get_taskalign_prompt(args, data, q_ID, demographic, wave, demographic_group, k=k)

    elif question_type=='task5':

        prompt = "In this task you will receive information on the distribution of responses from a group of people to related survey questions. Given this data, your task is to simulate an answer to a new question from the group. "
        prompt+= "First, I will provide the distribution of responses from the group of people to a series of questions. Afterwards, I will provide 5 example responses to the question to help you understand the formatting of this task. "

        if args.output_type=='sequence':
            prompt+= 'After the examples, please simulate 30 samples from the group of people for the new question asked. Please only respond with 30 multiple choice answers, no extra spaces, characters, quotes or text. Please only produce 30 characters. Answers with more than 30 characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'
        elif args.output_type=='model_logprobs': 
            prompt += 'After the examples, please simulate an answer from the group of people for the question asked. Please only respond with a single multiple choice answer, no extra spaces, characters, quotes or text. Please only produce 1 character. Answers with more than one characters will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'
        elif args.output_type=='express_distribution': 
            prompt += 'After the examples, please express the distribution of answers from the group of people for the question asked. Please only respond in the exact format of a dictionary mapping answer choice letter to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!'

        prompt+="\n\nGiven the fields 'context` and `question`, produce the fields `answer`. Your task will not have `context`.\n\n------\n\n"
        
        # ICL Is all other question in the wave 
        data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
        if wave=='Pew_American_Trends_Panel_disagreement_100': wave='Pew_American_Trends_Panel_disagreement_500'
        f = open('{}/{}/{}_data.json'.format(data_path, wave, demographic_group))
        icl_data = json.load(f)

        ICL_qIDS = get_ICL_qIDs(icl_data, q_ID, args.task3_type, wave, demographic, data_path)
        qID_to_wave = json.load(open('{}/qID_to_wave.json'.format(data_path)))
        for icl_qID in ICL_qIDS:
            if icl_qID != q_ID:
                if args.task3_type != 'all_but_one': 
                    ICL_qID_wave = qID_to_wave[q_ID]
                else: 
                    df_temp = pd.read_csv('{}/{}/info.csv'.format(data_path, wave))
                    ICL_qID_wave = df_temp.loc[df_temp['key'] == q_ID, 'survey']
                    ICL_qID_wave = ICL_qID_wave.iloc[0][4:]
                f = open('{}/{}/{}_data.json'.format(data_path, ICL_qID_wave, demographic_group))
                q_ID_data = json.load(f) # this is specific to the new wave 
                all_options, probs = [], []
                prompt += "\nContext: Let the probability that people from this group responds to the following question with "
                n = (sum(q_ID_data[icl_qID][demographic].values()))
                MC_options = list(q_ID_data[icl_qID][demographic].keys())
                if shuffled_incontext_labels: 
                    random.seed(random.randint(1, 1000))
                    random.shuffle(MC_options)
                for i, option in enumerate(MC_options):
                    all_options.append(options[i])
                    probs.append(q_ID_data[icl_qID][demographic][option]/n)
                    prompt +="{} be {}%, ".format(option, int((q_ID_data[icl_qID][demographic][option]/n)*100))
                prompt+= "\nQuestion: " + q_ID_data[icl_qID]['question_text'] + "?\n"
                for i, option in enumerate(MC_options):
                    # prompt +="{}. {}\n".format(options[i], option)
                    prompt +="{}. {}. ".format(options[i], option)
                if args.output_type=='sequence':
                    # Generate 30 flips
                    try: 
                        flips = random.choices(all_options, probs, k=30)
                        prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                    except: prompt+=''
                    
                elif args.output_type=='model_logprobs': 
                    try: 
                        flips = random.choices(all_options, probs, k=1)
                        prompt+="Answer: " + ' '.join(flips) + '\n\n------ '
                    except: prompt+=''
                    
                elif args.output_type=='express_distribution': 
                    prompt +="Answer: {"
                    for i, prob in enumerate(probs):
                        prompt+="'{}': '{}%', ".format(all_options[i], int(prob*100))
                    prompt = prompt[:-2] + '}\n\n------ ' # -2 to get rid of last space

        prompt+='\nYour turn! Please answer this question for the same group of people. As a reminder, this group is the exact same group as the group in the previous examples. The previous examples are used to provide an example of formatting and to give you insight into how this group would respond to such questions.\n'

        prompt+= "Question: " + data[q_ID]['question_text'] + "?\n"
        if model == 'llama3-70b' and args.output_type=='model_logprobs': 
            prompt=''
            for icl_qID in ICL_qIDS:
                if icl_qID != q_ID:
                    if args.task3_type != 'all_but_one': 
                        ICL_qID_wave = qID_to_wave[q_ID]
                    else: 
                        df_temp = pd.read_csv('{}/{}/info.csv'.format(data_path, wave))
                        ICL_qID_wave = df_temp.loc[df_temp['key'] == q_ID, 'survey']
                        ICL_qID_wave = ICL_qID_wave.iloc[0][4:]
                    f = open('{}/{}/{}_data.json'.format(data_path, ICL_qID_wave, demographic_group))
                    q_ID_data = json.load(f) # this is specific to the new wave 
                    all_options, probs = [], []
                    prompt += "\nLet the probability that a {} responds to the following question with ".format(demographic_in_prompt)
                    n = (sum(q_ID_data[icl_qID][demographic].values()))
                    MC_options = list(q_ID_data[icl_qID][demographic].keys())
                    if shuffled_incontext_labels: 
                        random.seed(random.randint(1, 1000))
                        random.shuffle(MC_options)
                    for i, option in enumerate(MC_options):
                        all_options.append(options[i])
                        probs.append(q_ID_data[icl_qID][demographic][option]/n)
                        prompt +="{} be {}%, ".format(option, int((q_ID_data[icl_qID][demographic][option]/n)*100))
                    prompt+= ": " + q_ID_data[icl_qID]['question_text'] + "?\n"
                    for i, option in enumerate(MC_options):
                        # prompt +="{}. {}\n".format(options[i], option)
                        prompt +="{}. {}. ".format(options[i], option)
                    
                    try: 
                        flips = random.choices(all_options, probs, k=1)
                        prompt+="Answer: " + ' '.join(flips) + '\n'
                    except: prompt+=''
            prompt+='Please answer this question for the group of {}s: '.format(demographic_in_prompt)
            prompt+=data[q_ID]['question_text'] + "?\n"
        for i, option in enumerate(list(data[q_ID][demographic].keys())):
            prompt +="{}. {}. ".format(options[i], option)
        prompt+="\nAnswer:"
    return prompt

def get_response(prompt, model, logprobs=False, top_logprobs=None, pipeline=None):
    if model=='gpt-3.5-turbo-0125' or model =='gpt-4': 
        try: 
            client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
                
            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                            {"role": "system", "content": "You are a helpful assistant."},
                                            {"role": "user", "content": prompt} # question that ask for a prompt 
                                        ],
                                logprobs=logprobs,
                                top_logprobs=top_logprobs,
                                temperature=1)
        except:
            try: response = client.chat.completions.create(
                                model=model,
                                messages=[
                                            {"role": "system", "content": "You are a helpful assistant."},
                                            {"role": "user", "content": prompt} # question that ask for a prompt 
                                        ],
                                logprobs=logprobs,
                                top_logprobs=top_logprobs,
                                temperature=1)
            except: print('open ai error')

    elif model=='anthropic_haiku':
        client = anthropic.Anthropic(
            # defaults to 
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

        response = client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=4096,
            temperature=1,
            system="You are a helpful assistant.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt}]}])

    elif model=='anthropic_opus':
        client = anthropic.Anthropic(
            # defaults to 
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

        try: 
            response = client.messages.create(
            model="claude-3-opus-20240229", max_tokens=4096,
            temperature=1,
            system="You are a helpful assistant.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt}]}])
        except:
            try: response = client.messages.create(
            model="claude-3-opus-20240229", max_tokens=4096,
            temperature=1,
            system="You are a helpful assistant.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt}]}])


            except: print('anthropic error')


    # elif model=='deepseek-coder-1.3b' or model=='deepseek-coder-6.7b' or model=='deepseek-coder-33b':
    #     tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/{}-base".format(model), trust_remote_code=True)

    #     # for deepseek, the variable pipeline contains the model 
    #     inputs = tokenizer(prompt, return_tensors="pt").to(pipeline.device)
    #     outputs = pipeline.generate(**inputs, max_new_tokens=128)
    #     response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]


    elif model=='llama3-8b' or model=='llama3-70b' or model=='phi-2' or model=='llama-2-7b' or model=='llama-2-13b' or model=='llama-2-70b' or model=='falcon-1b' or model=='falcon-7b' or model=='falcon-40b' or model=='falcon-180b': 
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(prompt,max_new_tokens=120, eos_token_id=terminators, do_sample=True, temperature=1, top_p=1)

        response = outputs[0]["generated_text"][len(prompt):]

    model_response=response
    print(response)
    if not logprobs:

        if model=='gpt-3.5-turbo-0125' or model=='gpt-4':
            try: response = model_response.choices[0].message.content.split('Answer: ')[1]
            except: response = model_response.choices[0].message.content# .split('Answer: ')

        elif model=='anthropic_haiku' or model=='anthropic_opus':
            try: response = model_response.content[0].text.split('Answer: ')[1]
            except: response = model_response.content[0].text

        elif  model=='llama3-8b' or model=='llama3-70b' or model=='phi-2' or model=='llama2-7b' or model=='llama2-13b' or model=='llama2-70b' or model=='deepseek-coder-1.3b' or model=='deepseek-coder-6.7b' or model=='deepseek-coder-33b' or model=='falcon-1b' or model=='falcon-7b' or model=='falcon-40b' or model=='falcon-180b': 
            try: response = model_response.split('Answer: ')[1]
            except: response = model_response
            print(response)

        return response, None
    else: 
        if model=='gpt-3.5-turbo-0125' or model=='gpt-4': 
            try: 
                response = model_response.choices[0].message.content.split("Answer: ")[1]
                logprobs = model_response.choices[0].logprobs.content[-1].top_logprobs

            except:
                response = model_response.choices[0].message.content
                logprobs = model_response.choices[0].logprobs.content[0].top_logprobs

        return response, logprobs
        
    return None, None


def get_sequence(prompt, model, pipeline=None):
    # print('hi2')
    response, _ = get_response(prompt, model, logprobs=False, top_logprobs=None, pipeline=pipeline)

    response = response.replace(" ", "")
    # if len(response)>30: 
    #     response = response[:30]


    return response


def extract_content_within_braces(input_string):
    # Define the regular expression pattern to match content within curly braces
    pattern = r'\{([^}]*)\}'
    
    # Find all matches of the pattern in the input string
    matches = re.findall(pattern, input_string)
    # Return the list of matches
    if matches: 
        return matches[0]
    else:
        return ""

def replace_double_quotes_with_single_quotes(input_string):
    # Define the regular expression pattern to match double quotes
    pattern = r'"'
    
    # Replace all occurrences of double quotes with single quotes
    replaced_string = re.sub(pattern, "'", input_string)
    
    return replaced_string


def add_single_quotes_around_characters(input_string):
    # Define the regular expression pattern to match characters in keys without quotes
    pattern = r'(\b\w+\b)(?=:)'  # Matches word characters (\w+) followed by a colon (:)

    # Replace each character in keys without quotes with single quotes around it
    replaced_string = re.sub(pattern, r"'\1'", input_string)

    return replaced_string


def get_expressed_distribution(prompt, model,pipeline=None):

    response_gpt, _ = get_response(prompt, model, logprobs=False, top_logprobs=None, pipeline=pipeline)
    return response_gpt

def llama3_logprobs(prompt, top_logprobs=20, pipeline=None):
    
    model_llama3, tokenizer_llama3 = pipeline
    
    # "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>".format(prompt)

    input_text = prompt
    # breakpoint()
    # input_text = "Question: Thinking about long-range foreign policy goals, how much priority, if any, do you think reducing illegal immigration into the US should be given? \nA. No priority. B. Refused. C. Some priority. D. Top priority. \nAnswer:"
    # input_text = "Question: What is the capital of France? \nA. Paris. B. Lyon. \n Answer:"
    # breakpoint()
    # Tokenize the input
    tokenizer_llama3.pad_token = tokenizer_llama3.eos_token

    inputs = tokenizer_llama3(input_text, return_tensors="pt", padding=True).to('cuda')
    # breakpoint()
    input_ids = inputs["input_ids"]
    # input_ids.to('cuda')
    attention_mask = inputs["attention_mask"]

    # Generate text with log probabilities
    output = model_llama3.generate(input_ids, max_length=len(input_ids[0])+10, output_scores=True, return_dict_in_generate=True, attention_mask=attention_mask,eos_token_id=tokenizer_llama3.eos_token_id)
        # do_sample=False,  # Greedy decoding for deterministic output
        # eos_token_id=tokenizer_llama3.eos_token_id  # Explicitly set EOS token
        # )

    # Extract the generated tokens and scores
    generated_ids = output.sequences
    logits = output.scores  # logits for each token in the output
    # Compute log probabilities
    log_probs = [torch.log_softmax(logit, dim=-1) for logit in logits]

    # Convert token IDs to text
    output_text = tokenizer_llama3.decode(generated_ids[0], skip_special_tokens=True)

    # Extract top 20 log probabilities and tokens for each word after the input text
    top_k = top_logprobs  # Number of top probabilities to extract
    start_index = len(input_ids[0])  # Start after the input text

    # Tokenize and clean up
    generated_tokens = tokenizer_llama3.convert_ids_to_tokens(generated_ids[0][start_index:])
    cleaned_generated_tokens = [token.replace('Ġ', '') for token in generated_tokens]

    top_log_probs = []
    top_tokens = []

    for i in range(len(log_probs)):
        top_values, top_indices = torch.topk(log_probs[i], top_k, dim=-1)
        top_log_probs.append(np.exp(top_values.cpu()))
        top_tokens.append(top_indices)

    # Convert top token indices to actual words
    top_tokens_text = [[tokenizer_llama3.decode([token]).strip() for token in top_token[0]] for top_token in top_tokens]
    
    token_to_prob = dict(zip(top_tokens_text[0], top_log_probs[0][0]))
    # final output: response, token_to_prob
    response = ''.join(cleaned_generated_tokens)
    return response, token_to_prob 

def get_model_logprob(prompt, model, pipeline):
    if model == 'llama3-70b': 
        response_gpt, logprobs = llama3_logprobs(prompt, 20, pipeline)
        token_to_prob = logprobs
    else:  
        response_gpt, logprobs = get_response(prompt, model, logprobs=True, top_logprobs=20, pipeline=pipeline)
        # create lowercase and stripped version of all and map it to log probs 
        token_to_prob = defaultdict(int)
        for i in range(20): token_to_prob[logprobs[i].token.strip().lower()]+= np.exp(logprobs[i].logprob) ## check this
        
    return response_gpt[0], token_to_prob

def calculate_proportions(response, MC_options, output_type):
    count, proportions = {}, {}
    if output_type=='sequence': 
        all_counts = 0
        for i, MC_option in enumerate(MC_options):
            all_counts += response.count(options[i]) + response.count(options[i].lower())
            count[options[i]] = response.count(options[i]) + response.count(options[i].lower())
        
        # Calculate proportions
        total_length = len(response)

        for i, _ in enumerate(MC_options): proportions[options[i]] = count[options[i]]/total_length
        return proportions

    elif output_type=='express_distribution': 
        # grab the thing in {} out of the reponses
        # response = extract_content_within_braces(response)
        # replace all " with '
        response = replace_double_quotes_with_single_quotes(response)
        # make sure all the keys are enclosed in single quotes
        response = add_single_quotes_around_characters(response)

        try: proportions=eval(response)
        except: 
            # breakpoint()
            print('answer dict formatting issue')
            return None

        # conver the percentages into proportions
        for MC_option in proportions.keys():
            try: proportions[MC_option] = int(float(proportions[MC_option][:-1]))*0.01
            except: proportions[MC_option]=0
        return proportions


def task_disagree500(q_IDS, demographic, data_path, model, args, waves, demographic_group, question_type='task_1', k=1, ficticious_group_ablation=False, shuffled_incontext_labels=False, pipeline=None):

    
    # make folder if it doesnt exist
    folder_path = '{}/results/{}/'.format(os.getcwd(), args.dataset)
    if not os.path.exists(folder_path): os.makedirs(folder_path)

    folder_path += '{}/'.format(args.output_type)
    if not os.path.exists(folder_path): os.makedirs(folder_path)

    folder_path += '{}/'.format(model)
    if not os.path.exists(folder_path): os.makedirs(folder_path)

    if question_type == 'task3': 
        # folder_path += '{}/'.format(args.task3_type)
        # if not os.path.exists(folder_path): os.makedirs(folder_path)

        folder_path += '{}/'.format(question_type+"_{}".format(args.task3_type))
        if not os.path.exists(folder_path): os.makedirs(folder_path)

    else: 
        folder_path += '{}/'.format(question_type)
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        
    if args.dataset=='opinionqa':
        wave_name = args.wave
        folder_path += '{}/'.format(wave_name)
        if not os.path.exists(folder_path): os.makedirs(folder_path)
    folder_path += '{}'.format(demographic_group)
    if not os.path.exists(folder_path): os.makedirs(folder_path)

    if os.path.exists(folder_path+ '/{}.json'.format(demographic)): 
        with open(folder_path+ '/{}.json'.format(demographic), 'r') as file: results_per_q = json.load(file)
    
    else: results_per_q = {} 
    

    if args.dataset=='opinionqa': data = json.load(open(data_path + args.wave + '/' + demographic_group + "_data.json"))
    elif args.dataset=='nytimes': data = json.load(open(data_path +'/' + demographic_group + "_data.json"))
    elif args.dataset=='globalvalues': data = json.load(open(data_path +'/' + demographic + "_data.json"))
    else: data = {}

    for i, q_ID in enumerate(q_IDS):
        
        
        # print(q_ID)
        if q_ID not in results_per_q.keys(): 

            expected_results = {} 

            if args.dataset=='opinionqa': MC_options = list(data[q_ID][demographic].keys())
            elif args.dataset=='nytimes':  MC_options = data[q_ID]['MC_options']
            elif args.dataset=='globalvalues': 
                data[q_ID]['options'] = ast.literal_eval(data[q_ID]['options'])
                MC_options = data[q_ID]['options']

            if args.dataset=='opinionqa':
                wave = waves[i][4:]
                n = (sum(data[q_ID][demographic].values()))
                for j, MC_option in enumerate(MC_options):
                    expected_results[options[j]] = data[q_ID][demographic][MC_option]/n
                
                task_prompt = get_prompt_opinionqa(args, model, question_type, data, q_ID, demographic, args.wave, demographic_group, k=k, ficticious_group_ablation=ficticious_group_ablation, shuffled_incontext_labels=shuffled_incontext_labels)
                
            elif args.dataset=='nytimes': 
                
                n = (sum(data[q_ID][demographic].values()))
                for j, MC_option in enumerate(MC_options): 
                    if str(MC_option) in data[q_ID][demographic]: 
                        expected_results[options[j]] = data[q_ID][demographic][str(MC_option)]/n
                    else: expected_results[options[j]] = 0
   

                task_prompt = get_prompt_nytimes(args, model, question_type, data, q_ID, demographic, args.wave, demographic_group, k=k, ficticious_group_ablation=ficticious_group_ablation, shuffled_incontext_labels=shuffled_incontext_labels)
                

            elif args.dataset=='globalvalues': 
                for j, MC_option in enumerate(MC_options): 
                    expected_results[options[j]] = data[q_ID]['data'][j]

                task_prompt = get_prompt_globalvalues(args, question_type, data, q_ID, demographic, args.wave, demographic_group, k=k, ficticious_group_ablation=ficticious_group_ablation, shuffled_incontext_labels=shuffled_incontext_labels)
                

            actual_results = {key: [] for key in expected_results}
            responses = []
            
            if args.output_type=='sequence': 
                
                for _ in range(args.n_seq):
                    response = get_sequence(task_prompt, model, pipeline)
                    proportions = calculate_proportions(response, MC_options, output_type=args.output_type)
                    if proportions: 
                        for mc in proportions.keys():
                            actual_results[mc].append(proportions[mc])
                    responses.append(response)

            elif args.output_type=='model_logprobs':
                for _ in range(args.n_sample):
                    response, token_to_prob = get_model_logprob(task_prompt, model, pipeline)
                    for token in list(token_to_prob.keys()): 
                        token = token.strip()# .lower()
                        logic = False
                        for answer_choice in list(expected_results.keys()): logic = logic or (token == answer_choice)
                        
                        if logic: actual_results[token.upper()].append(token_to_prob[token].item())
                    responses.append(response)


            elif args.output_type=='express_distribution':
                for _ in range(args.n_seq):
                    response = get_expressed_distribution(task_prompt, model, pipeline)
                    proportions = calculate_proportions(response, MC_options, output_type=args.output_type)
                    if proportions:
                        for mc in actual_results.keys():
                            if mc in proportions.keys():
                                actual_results[mc].append(proportions[mc])

                    responses.append(response)

            avg_actual_results, std_actual_results = {}, {}
            for key in actual_results.keys():
                if actual_results[key]: 
                    avg = np.mean(actual_results[key])
                    std = np.std(actual_results[key])
                else: 
                    actual_results[key] = list(np.zeros(args.n_seq))
                    avg = 0
                    std = 0
                avg_actual_results[key] = avg
                std_actual_results[key] = std

            # actual results: model's output probs
            # expected results: opinionQA GT
            # breakpoint()
        
            results = {'actual_results': actual_results, 'avg_actual_results': avg_actual_results, 'std_actual_results': std_actual_results, 'response': responses}
            if question_type=='task0':
                if args.dataset=='opinionqa':
                    for demographic_group_temp in ['POLPARTY', 'SEX', 'CREGION', 'EDUCATION', 'INCOME', 'RACE']:
                        data_temp = json.load(open(data_path + args.wave + '/' + demographic_group_temp + "_data.json"))

                        for demographic_temp in dem_group_to_dem_mapping[demographic_group_temp]:
                            expected_results = {} 
                            n = (sum(data_temp[q_ID][demographic_temp].values()))
                            for j, MC_option in enumerate(data_temp[q_ID][demographic_temp]):
                                expected_results[options[j]] = data_temp[q_ID][demographic_temp][MC_option]/n
                            results['expected_results_{}_{}'.format(demographic_group_temp, demographic_temp)] = expected_results
                elif args.dataset=='nytimes':
                    for demographic_group_temp in ['POLPARTY', 'SEX']:
                        data_temp = json.load(open(data_path +'/' + demographic_group_temp + "_data.json"))

                        for demographic_temp in dem_group_to_dem_mapping[demographic_group_temp]:
                            expected_results = {} 

                            n = (sum(data_temp[q_ID][demographic_temp].values()))
                            for j, MC_option in enumerate(data_temp[q_ID][demographic_temp]):
                                expected_results[options[j]] = data_temp[q_ID][demographic_temp][str(MC_option)]/n
                            results['expected_results_{}_{}'.format(demographic_group_temp, demographic_temp)] = expected_results
                elif args.dataset=='globalvalues':
                    results['expected_results'] = expected_results

            else: 
                results['expected_results'] = expected_results
            # print(expected_results, results['avg_actual_results'])

            # need to calculate sample probabilities too
            if args.output_type=='model_logprobs': 
                tokens, counts = np.unique(responses, return_counts=True)
                token_to_count = dict(zip(tokens, counts))
                results['actual_results_sampled'] = {}
                for key in actual_results.keys():
                    if key in token_to_count.keys(): results['actual_results_sampled'][key] = token_to_count[key]/args.n_sample
                    else: results['actual_results_sampled'][key] = 0

            results_per_q[q_ID] = results
            if ficticious_group_ablation:                
                with open(folder_path + '/{}.json'.format(ficticious_group_ablation_mapping[demographic]), 'w') as f:
                    json.dump(results_per_q, f)
            
            elif shuffled_incontext_labels:
                with open(folder_path + '/{}_{}.json'.format(demographic, 'shuffled'), 'w') as f:
                    json.dump(results_per_q, f)

            elif question_type=='task4' or question_type=='align':                
                with open(folder_path + '/{}_{}.json'.format(demographic, k), 'w') as f:
                    json.dump(results_per_q, f)

            else: 
                with open(folder_path + '/{}.json'.format(demographic), 'w') as f:
                    json.dump(results_per_q, f)

            # breakpoint()
    return


def extract_boxed_text(text):
    # Define the regular expression pattern
    pattern = r'\\boxed\{(.*?)\}'
    
    # Use re.findall to extract the boxed text
    boxed_texts = re.findall(pattern, text)
    
    # Return the extracted text
    return boxed_texts

def process_dataset(dataset_name):
    save_data_path = "{}/results/{}/training_data.json".format(os.getcwd(), dataset_name)

    if os.path.exists(save_data_path):
        with open(save_data_path, 'r') as file:
            q_a = json.load(file)
            return q_a

    else: 
        q_a = {}
        if dataset_name=='MATH':
            path = '{}/MATH/train/counting_and_probability'
            for json_file in os.listdir(path):
                with open(path+'/'+json_file, "r") as jf:
                    data = json.load(jf)
                    if data['level']=="Level 4" or data['level']=="Level 5":
                        if 'frac' not in extract_boxed_text(data['solution'])[0]: 
                            q_a[data['problem']] = extract_boxed_text(data['solution'])[0]

        if dataset_name=='AQuA':
            path = '{}/AQuA/train.txt'
            data = []
            with open(path) as f:
                for line in f:
                    data = json.loads(line)
                    if data['options'][option_to_num[data['correct']]][2:].isnumeric():
                        q_a[data['question']] = {'options': data['options'], 'options': data['options'] , 'correct_option': data['correct'], 'correct_answer': data['options'][option_to_num[data['correct']]][2:] }
        
        # if dataset_name=='gsm8k':
        #     dataset = load_dataset("parquet", data_files={'train': '{}/gsm8k/train-00000-of-00001.parquet'})
        #     for i, d in enumerate(dataset['train']): 
        #         if i < 200: 
        #             question = d['question']
        #             answer = d['answer'].split('\n')[-1].split('#### ')[1]
        #             q_a[question] = answer


        with open(save_data_path, "w") as json_file:
            json.dump(q_a, json_file)

        return q_a


def calc_total_variation(p, q):
    # try: assert(np.sum(p)<1.01 and np.sum(p)>0.99)
    # except: print('sum(p): {}'.format(np.sum(p)))
    
    # try: assert(np.sum(q)<1.01 and np.sum(q)>0.99)
    # except: print('sum(q): {}'.format(np.sum(q)))

    p=np.array(p)
    q=np.array(q)
    return 0.5* np.sum(np.abs(p-q))

def kl_divergence(p, q):
    a = np.asarray(p, dtype=float)
    b = np.asarray(q, dtype=float)
    return np.sum(np.where(((a != 0) & (b!=0)), a * np.log(a / b), 0))

def calc_jsd(p, q):
    return distance.jensenshannon(p,q)**2

def calc_wasserstein(p,q):
    return wasserstein_distance(p,q)



def compute_tv_GT(task='task1', model='gpt-4', demographic_group = 'POLPARTY', demographic='Democrat', wave=None, k=None, output_type='sequence', dataset='opinionqa'):
    print(task, demographic)
    # print(demographic_group, demographic, wave)
    # for task 0: get data from the NONE json and get the entry with the demographic  
    if dataset=='opinionqa':
        if task=='task0': 
            path = '{}/results/opinionqa/{}/{}/{}/{}/{}'.format(os.getcwd(), output_type, model, task, wave, 'NONE')
            with open(path + '/{}.json'.format('Democrat'), 'r') as file: data = json.load(file)

        else: 
            path = '{}/results/opinionqa/{}/{}/{}/{}/{}'.format(os.getcwd(), output_type, model, task, wave, demographic_group)
            with open(path + '/{}.json'.format(demographic), 'r') as file: data = json.load(file)

    elif dataset=='nytimes':
        if task=='task0': 
            path = '{}/results/{}/{}/{}/{}/{}'.format(os.getcwd(), dataset, output_type, model, task, 'NONE')
            with open(path + '/{}.json'.format('Democrat'), 'r') as file: data = json.load(file)
        else: 
            path = '{}/results/{}/{}/{}/{}/{}'.format(os.getcwd(), dataset, output_type, model, task, demographic_group)
            with open(path + '/{}.json'.format(demographic), 'r') as file: data = json.load(file)


    elif dataset=='nytimes':
        if task=='task0': 
            path = '{}/results/{}/{}/{}/{}/{}'.format(os.getcwd(), dataset, output_type, model, task, 'NONE')
            with open(path + '/{}.json'.format('Democrat'), 'r') as file: data = json.load(file)
        else: 
            path = '{}/results/{}/{}/{}/{}/{}'.format(os.getcwd(), dataset, output_type, model, task, demographic_group)
            with open(path + '/{}.json'.format(demographic), 'r') as file: data = json.load(file)


 
    elif dataset=='globalvalues':
        path = '{}/results/{}/{}/{}/{}/{}'.format(os.getcwd(), dataset, output_type, model, task, demographic_group)
        with open(path + '/{}.json'.format(demographic), 'r') as file: data = json.load(file)
    
        


    all_tvs, all_jsds, all_wassersteins = [], [], []
    print(len(list(data.keys())))

    for i, question in enumerate(list(data.keys())):
        all_tvs_i, all_jsds_i, all_wassersteins_i = [], [], []

        # actual_results = list(data[question]['avg_actual_results'].values())
        actual_results = data[question]['actual_results']
        
        # if uniform: 
        #     len_actual_results = len(actual_results.keys())
        #     for key in list(actual_results.keys()): actual_results[key]=1/len_actual_results
        #     actual_results =list(actual_results.values())
        if actual_results != {} and actual_results != [] : 
            if task=='task0' and dataset=='opinionqa': 
                data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
                expected_results = {} 
                data_temp = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
                try: n = (sum(data_temp[question][demographic].values()))
                except: breakpoint()
                for j, MC_option in enumerate(data_temp[question]['MC_options']):
                    if options[j] in data[question]['avg_actual_results'].keys():
                        if MC_option in data_temp[question][demographic].keys(): 
                            expected_results[options[j]] = data_temp[question][demographic][MC_option]/n
                        else: expected_results[options[j]] = 0
                expected_results = list(expected_results.values())
            

            elif task=='task0' and dataset=='nytimes':
                expected_results = list(data[question]['expected_results_{}_{}'.format(demographic_group, demographic)].values())

            else: 
                expected_results = list(data[question]['expected_results'].values())

            # COMPUTE NEW ACTUAL RESULTS based on the expected results 
            probs = expected_results
            if np.sum(probs)==0: break
            all_options = list(data[question]['actual_results'].keys())
            for j in range(1000):
                if j%100 == 0: print(j)
                try: flips = random.choices(all_options, probs, k=30)
                except: breakpoint()
                simulated_response = ''.join(flips)
                proportions = calculate_proportions(simulated_response, all_options, output_type='sequence')
                actual_results = list(proportions.values())
            
                try: 
                    tv = calc_total_variation(actual_results, expected_results)
                    jsd = calc_jsd(actual_results, expected_results)
                    wasserstein = calc_wasserstein(actual_results, expected_results)
                except: breakpoint()

                all_tvs_i.append(tv)
                all_jsds_i.append(jsd)
                all_wassersteins_i.append(wasserstein)
        all_tvs.append(np.mean(all_tvs_i))
        all_jsds.append(np.mean(all_jsds_i))
        all_wassersteins.append(np.mean(all_wassersteins_i))
    TV, JSD, WASSERSTEIN = np.mean(all_tvs), np.mean(all_jsds), np.mean(all_wassersteins)
    return TV, JSD, WASSERSTEIN, all_tvs

### TO DO: make this eval slightly different for task 0 where
# make this for a specific demographci in the demographic group 
def compute_tv(task='task1', model='gpt-4', demographic_group = 'POLPARTY', dataset='opinionqa', demographic='Democrat', wave = 'American_Trends_Panel_W26', k=None, output_type='sequence',ficticious_group_ablation=False, shuffled_incontext_labels=False, uniform=False, LB1=False, LB2=False):
    print(task, demographic)
    # print(demographic_group, demographic, wave)
    # for task 0: get data from the NONE json and get the entry with the demographic  
    if dataset=='opinionqa':
        if task=='task0': 
            path = '{}/results/opinionqa/{}/{}/{}/{}/{}'.format(os.getcwd(), output_type, model, task, wave, 'NONE')
            with open(path + '/{}.json'.format('Democrat'), 'r') as file: data = json.load(file)

        else: 
            path = '{}/results/opinionqa/{}/{}/{}/{}/{}'.format(os.getcwd(), output_type, model, task, wave, demographic_group)
            with open(path + '/{}.json'.format(demographic), 'r') as file: data = json.load(file)

    elif dataset=='nytimes':
        if task=='task0': 
            path = '{}/results/{}/{}/{}/{}/{}'.format(os.getcwd(), dataset, output_type, model, task, 'NONE')
            with open(path + '/{}.json'.format('Democrat'), 'r') as file: data = json.load(file)
        else: 
            path = '{}/results/{}/{}/{}/{}/{}'.format(os.getcwd(), dataset, output_type, model, task, demographic_group)
            with open(path + '/{}.json'.format(demographic), 'r') as file: data = json.load(file)

    elif dataset=='globalvalues':
        path = '{}/results/{}/{}/{}/{}/{}'.format(os.getcwd(), dataset, output_type, model, task, demographic_group)
        with open(path + '/{}.json'.format(demographic), 'r') as file: data = json.load(file)
    
        
    print(len(list(data.keys())))
    all_tvs, all_jsds, all_wassersteins = [], [], []
    all_tv_dict = {}

    for i, question in enumerate(list(data.keys())):

        # actual_results = list(data[question]['avg_actual_results'].values())
        actual_results = list(data[question]['avg_actual_results'].values())

        if uniform: 
            actual_results = data[question]['actual_results']
            len_actual_results = len(actual_results.keys())
            for key in list(actual_results.keys()): actual_results[key]=1/len_actual_results
            actual_results =list(actual_results.values())
            if dataset=='nytimes': assert(len_actual_results==4)
            
        
        if LB1: 
            actual_results = data[question]['actual_results']
            # get the key with the highest value
            max_key = max(actual_results, key=actual_results.get)
            # set all values to 0 except for the max key which is set to 1  
            for key in list(actual_results.keys()): actual_results[key]=0
            actual_results[max_key]=1
            actual_results =list(actual_results.values())

        if LB2: 
            actual_results = data[question]['actual_results']
            # get the key with the highest value
            min_key = min(actual_results, key=actual_results.get)
            # set all values to 0 except for the max key which is set to 1  
            for key in list(actual_results.keys()): actual_results[key]=0
            actual_results[min_key]=1
            actual_results =list(actual_results.values())
            
        if actual_results != {} and actual_results != [] : 
            if task=='task0' and dataset=='opinionqa':
                data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
                expected_results = {} 
                data_temp = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
                try: n = (sum(data_temp[question][demographic].values()))
                except: breakpoint()
                for j, MC_option in enumerate(data_temp[question]['MC_options']):
                    if options[j] in data[question]['avg_actual_results'].keys():
                        if MC_option in data_temp[question][demographic].keys(): 
                            expected_results[options[j]] = data_temp[question][demographic][MC_option]/n
                        else: expected_results[options[j]] = 0
                expected_results = list(expected_results.values())
            elif task=='task0' and dataset=='nytimes':
                expected_results = list(data[question]['expected_results_{}_{}'.format(demographic_group, demographic)].values())
            
            else: expected_results = list(data[question]['expected_results'].values())
            # breakpoint()
            try: 
                tv = calc_total_variation(actual_results, expected_results)
                jsd = calc_jsd(actual_results, expected_results)
                wasserstein = calc_wasserstein(actual_results, expected_results)
            except: tv, jsd, wasserstein = np.nan, np.nan, np.nan

            all_tvs.append(tv)
            all_tv_dict[question]=tv
            all_jsds.append(jsd)
            all_wassersteins.append(wasserstein)


    all_tvs, all_jsds, all_wassersteins =  np.array(all_tvs), np.array(all_jsds), np.array(all_wassersteins)
    TV, JSD, WASSERSTEIN = np.nanmean(all_tvs), np.nanmean(all_jsds), np.nanmean(all_wassersteins)
    
    return TV, JSD, WASSERSTEIN, all_tvs, all_tv_dict


def compute_tv_per_question(task='task1', model='gpt-4', demographic_group = 'POLPARTY', demographic='Democrat', wave = 'American_Trends_Panel_W26', q_ID = None, output_type='sequence'):
    # print(demographic_group, demographic, wave)
    # for task 0: get data from the NONE json and get the entry with the demographic  
    if task=='task0': 
        path = '{}/results/opinionqa/{}/{}/{}/{}/{}'.format(os.getcwd(), output_type, model, task, wave, 'NONE')
        with open(path + '/{}.json'.format('Democrat'), 'r') as file: data = json.load(file)

    else: 
        path = '{}/results/opinionqa/{}/{}/{}/{}/{}'.format(os.getcwd(), output_type, model, task, wave, demographic_group)
        with open(path + '/{}.json'.format(demographic), 'r') as file: data = json.load(file)

    question = q_ID
    
    actual_results = list(data[question]['avg_actual_results'].values())
    if task=='task0': 
        data_path = '{}/opinions_qa/data/human_resp/'.format(os.getcwd())
        expected_results = {} 
        data_temp = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
        try: n = (sum(data_temp[question][demographic].values()))
        except: breakpoint()
        for j, MC_option in enumerate(data_temp[question]['MC_options']):
            if options[j] in data[question]['avg_actual_results'].keys():
                if MC_option in data_temp[question][demographic].keys(): 
                    expected_results[options[j]] = data_temp[question][demographic][MC_option]/n
                else: expected_results[options[j]] = 0
        expected_results = list(expected_results.values())

    else: expected_results = list(data[question]['expected_results'].values())
    
    tv = calc_total_variation(actual_results, expected_results)
    jsd = calc_jsd(actual_results, expected_results)
    wasserstein = calc_wasserstein(actual_results, expected_results)

    return tv, jsd, wasserstein


def plot(task='task1', model='gpt-4', demographic_group = 'POLPARTY', wave = 'American_Trends_Panel_W26', k=None, output_type = 'sequence', ficticious_group_ablation=False, shuffled_incontext_labels=False):
    # print(demographic_group, demographic, wave)
    # for task 0: get data from the NONE json and get the entry with the demographic

    for demographic in dem_group_to_dem_mapping[demographic_group]:
        print(demographic)
        if task=='task0': 
            path = '{}/results/opinionqa/{}/{}/{}/{}/{}'.format(os.getcwd(), output_type, model, task, wave, 'NONE')
            with open(path + '/{}.json'.format('Democrat'), 'r') as file: data = json.load(file)

        else: 
            path = '{}/results/opinionqa/{}/{}/{}/{}/{}'.format(os.getcwd(), output_type, model, task, wave, demographic_group)
            with open(path + '/{}.json'.format(demographic), 'r') as file: data = json.load(file)


        expected, actual_all, expected_all, actual_mean, actual_std = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        incontext_mean, incontext_std = defaultdict(list), defaultdict(list)
        first = True

        for i, q_ID in enumerate(list(data.keys())):

            # incontext_all = defaultdict(list)

            if task=='task0': expected_results = data[q_ID]['expected_results_{}_{}'.format(demographic_group, demographic)]
            else: expected_results = data[q_ID]['expected_results']
            actual_results = data[q_ID]['actual_results']
            
            avg_actual_results = data[q_ID]['avg_actual_results']
            std_actual_results = data[q_ID]['std_actual_results']

            # for key in list(expected_results.keys()):
            #     expected_results[key] = list(expected_results[key] * np.ones(len(actual_results[key])))
            
            # data_path = '/Users/nicolemeister/Desktop/STANFORD/distributions/opinions_qa/data/human_resp/'
            # pew_data = json.load(open(data_path + wave + '/' + demographic_group + "_data.json"))
            # f = open(data_path + wave + '/question_similarity.json')
            # sim_data = json.load(f)
            # similar_qIDS = sim_data[q_ID]['similar_questionIDs']
            # for similar_qID in similar_qIDS:
            #     if similar_qID != q_ID:
            #         # only add this question if the number of MC choices in the similar questions matches the number of MC options in the qID
            #         if len(pew_data[q_ID][demographic].keys()) == len(pew_data[similar_qID][demographic].keys()):
            #             n = (sum(pew_data[similar_qID][demographic].values()))
            #             for i, option in enumerate(list(pew_data[similar_qID][demographic].keys())):
            #             incontext_all[options[i]].append(pew_data[similar_qID][demographic][option]/n)
            
            # # compute incontext mean
            # for key in list(incontext_all.keys()):
            #     incontext_mean[key].append(np.mean(incontext_all[key]))
            #     incontext_std[key].append(np.std(incontext_all[key]))
            #     expected[key].append(expected_results[key][0])
            #     actual_mean[key].append(avg_actual_results[key])
            #     actual_std[key].append(std_actual_results[key])

            for key in list(expected_results.keys()):
                if first: 
                    if ficticious_group_ablation: plt.scatter(expected_results[key], avg_actual_results[key], alpha=0.2, c=dem_to_color[demographic], label=ficticious_group_ablation_mapping[demographic] , marker='^') 
                    else: 
                    
                        plt.scatter(expected_results[key], avg_actual_results[key], alpha=0.2, c=dem_to_color[demographic], label=demographic , marker='^') 
                        # plt.scatter(expected[key], np.array(actual_mean[key])-np.array(incontext_mean[key]), alpha=0.2,  c='b', label='sequence-incontext')
                        first = False
                else: 
                    # plt.scatter(expected[key], np.array(actual_mean[key])-np.array(incontext_mean[key]), alpha=0.2,  c='b')
                    # plt.errorbar(expected[key], incontext_mean[key]-actual_mean[key], incontext_std[key], alpha=0.2, linestyle='None', c='r', marker='^')
                    plt.errorbar(expected_results[key], avg_actual_results[key], std_actual_results[key], alpha=0.2, linestyle='None', c=dem_to_color[demographic], marker='^') 

    TV = compute_tv(task=task, model=model, demographic_group=demographic_group, demographic=demographic, wave=wave, output_type=output_type)

    if ficticious_group_ablation: plt.title("{}, {}, {}, {}, {:.4f}".format("Task {}".format(task[-1]), model, wave, "Ficticious Group"))
    elif shuffled_incontext_labels: plt.title("{}, {}, {} (Shuffled MC), {}\n TV: {:.4f}".format("Task {}".format(task[-1]), model, wave, demographic_group, TV))
    elif k: plt.title("{}, k={}: {}, {}, {}, {:.4f}".format("Task {}".format(task[-1]), k, model, wave, demographic_group, TV))
    else: plt.title("{}: {}, {}, {}, {:.4f}".format("Task {}".format(task[-1]), model, wave, demographic_group, TV))

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    x = np.linspace(0, 1, 100)
    # Plot y=x as a red dashed line
    plt.plot(x, x, 'r--', label='y=x', alpha=0.7)
    plt.xlabel("Expected Probability of MC Option")
    plt.ylabel("Sequence Probability")
    plt.legend()
    
    if task=='task0': folder_path = '{}/results/opinionqa/{}/{}/{}/{}/{}'.format(os.getcwd(), output_type, model, task, wave, 'NONE')
    else: folder_path = '{}/results/opinionqa/{}/{}/{}/{}/{}'.format(os.getcwd(), output_type, model, task, wave, demographic_group)

    plt.savefig('{}/{}_plot.png'.format(folder_path, demographic_group))
    plt.show()

    return