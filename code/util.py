from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
import pandas as pd


"""
Calculate directness of stance

from https://arxiv.org/pdf/1605.01655.pdf "Properties 3 and 4 are addressed to some extent by the fact that removing the query hashtag can sometimes result in tweetsthat do not explicitly mention the target."

and "We removed the query hashtags from the tweets to exclude obvious cues for the classification task."

"""

def directed(tweet, target):
    target_words = target.lower().split(' ')
    if target == 'Recep Tayyip Erdoğan':
        target_words.append('erdogan')  
    for target_word in target_words:
        if target_word in tweet.lower():
            return 'Direct'
    return 'Indirect'

def data_save_semeval():
	return

def data_load_and_save(datasets = 'all', test_size = 0.7, runs = 1):
	if datasets != 'all':
		all_data = data_load(datasets = datasets)
	else:
		all_data = pd.read_csv("../data/all_data.csv", sep = "\t")

	print(len(all_data))

	# for each target split data into train and test, recombine and save
	training_sets = {}
	test_sets = {}
	training_set = {}
	test_set = {}
	targets = all_data['Target'].unique()
	for run in range(runs):
		for target in targets:
			if target not in training_sets.keys():
				training_set[target] = None
				test_set[target] = None
				training_sets[target] = []
				test_sets[target] = []
			data = all_data[all_data['Target'] == target]
			training_set[target], test_set[target] = train_test_split(data, test_size = len(data) - test_size, stratify = data['Stance'])
			training_sets[target].append(training_set[target])
			test_sets[target].append(test_set[target])

		# recombine and save
		all_training_data = pd.DataFrame()
		all_test_data = pd.DataFrame()
		for target in targets:
			all_training_data = all_training_data.append(training_set[target])
			all_test_data = all_test_data.append(test_set[target])

		print(len(all_training_data))
		print(len(all_test_data))

		all_training_data.to_csv("../data/train_test_split/all_train_%d.csv" %(run), sep = "\t", index = False)
		all_test_data.to_csv("../data/train_test_split/all_test_%d.csv" %(run), sep = "\t", index = False)



	return training_sets, test_sets


def get_results(cr, y_true, y_pred, method, directed = 'Both', dataset = 'All', target = 'All', none = True):
    #print(cr)
    total = 0
    if none:
        total = (cr['AGAINST']['support'] + cr['FAVOR']['support'] + cr['NONE']['support'])
    else:
        total = (cr['AGAINST']['support'] + cr['FAVOR']['support'])
    
    result = {}
    result['Method'] = method
    result['Directed'] = directed
    result['Dataset'] = dataset
    result['Target'] = target
    
    result['Fraction of Against Class'] = float(cr['AGAINST']['support'])/total
    result['Against Class Precision'] = cr['AGAINST']['precision']
    result['Against Class Recall'] = cr['AGAINST']['recall']
    result['Against Class F1'] = cr['AGAINST']['f1-score']
    result['Fraction of Predicted Against'] = len([i for i in y_pred if i == 'AGAINST'])/len(y_pred)
    
    result['Fraction of Favor Class'] = float(cr['FAVOR']['support'])/total
    result['Favor Class Precision'] = cr['FAVOR']['precision']
    result['Favor Class Recall'] = cr['FAVOR']['recall']
    result['Favor Class F1'] = cr['FAVOR']['f1-score']
    result['Fraction of Predicted Favor'] = len([i for i in y_pred if i == 'FAVOR'])/len(y_pred)

    
    if none:
        result['Fraction of None Class'] = float(cr['NONE']['support'])/total
        result['None Class Precision'] = cr['NONE']['precision']
        result['None Class Recall'] = cr['NONE']['recall']
        result['None Class F1'] = cr['NONE']['f1-score']
        result['Fraction of Predicted None'] = len([i for i in y_pred if i == 'NONE'])/len(y_pred)
    
    
    result['Macro Average Precision'] = cr['macro avg']['precision']
    result['Macro Average Recall'] = cr['macro avg']['recall']
    result['Weighted F1'] = f1_score(y_true, y_pred, average = 'weighted')
    result['Macro F1'] = cr['macro avg']['f1-score']
    
    result['Semeval F1'] = (cr['FAVOR']['f1-score'] + cr['AGAINST']['f1-score'])/2
    
    
    return result


# function for merging datasets; no longer required
def data_load(datasets = ['semevala', 'semevalb', 'constance', 'mtsd', 'presidents'], hrc_only = True, test_size = 0.7):
	
	datafiles = {'semevala': "../../data_emnlp/semeval/SemEval2016-Task6-subtaskA-testdata-gold_cleaned.txt",
				 'semevalb' : "../../data_emnlp/semeval/SemEval2016-Task6-subtaskB-testdata-gold_cleaned.txt",
				 'mtsd' : "../../data_emnlp/mtsd/multi_stance_data_trump_clinton_SE.csv",
				 'constance' : "../../data_emnlp/constance/constance_sd_trump_clinton_tweets.csv",
				 'f8' : "../../data_emnlp/F8/F8_semeval_format.csv",
				 'presidents' : "../../data_emnlp/presidents/twitter_titling_corpus_hydrated_extended.csv",
	}

	all_data = pd.DataFrame()
	for dataset in datasets:
		# if dataset == 'constance':
		# 	data = pd.read_csv(datafiles[dataset], sep = ',')
		if dataset != 'presidents':
			data = pd.read_csv(datafiles[dataset], sep = "\t")
		else:
			data = pd.read_csv(datafiles[dataset], sep = "\t")
			data.columns = ['extra', 'ID', 'Target','country','Stance','naming_form','Tweet']
			data['Stance'] = data['Stance'].map({-1 : 'AGAINST', 1 : 'FAVOR', 0 : 'NONE'})
			data = data[data['Tweet'] != 'Missing']

		if dataset == 'mtsd':
			data['Target'] = ['Hillary Clinton' if i == 'Hilary Clinton' else i for i in data['Target']]
		# print(dataset)
		# print(data.head())
		# print(len(data))
		# print(data.isna().sum())
		# print()
		data = data[['ID', 'Target', 'Tweet', 'Stance']]
		data['Dataset'] = dataset
		if dataset == 'semevala' and hrc_only:
			data = data[data['Target'] == 'Hillary Clinton']
		all_data = all_data.append(data)
		all_data['Stance'] = all_data['Stance'].replace({'NO STANCE' : 'NONE'})

		all_data = all_data.dropna()

	all_data['ID'] = range(0, len(all_data))
	all_data.to_csv("../data/all_data.csv", sep = "\t", index = False)
	all_data[['ID', 'Target', 'Stance', 'Dataset']].to_csv("../data/all_data_no_tweet_text.csv", sep = "\t", index = False) # to be made publically available.
	return all_data

