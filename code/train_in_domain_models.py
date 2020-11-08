import time
import datetime
from util import data_load_and_save

def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	
	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))

import sys

import pandas as pd
import numpy as np

import re
from nltk.corpus import stopwords # Import the stop word list

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


import pickle

from sklearn.metrics import classification_report


def tweet_to_words( raw_tweet ):
	# Function to convert a raw tweet to a string of words
	# The input is a single string (a raw movie tweet), and 
	# the output is a single string (a preprocessed movie tweet)
	#
	tweet_text = raw_tweet.replace("#", '')
	tweet_text = tweet_text.replace("@", '')
	# 2. Remove non-letters		
	letters_only = re.sub("[^a-zA-Z]", " ", tweet_text) 
	#
	# 3. Convert to lower case, split into individual words
	words = letters_only.lower().split()							 
	#
	# 4. In Python, searching a set is much faster than searching
	#   a list, so convert the stop words to a set
	stops = set(stopwords.words("english"))				  
	# 
	# 5. Remove stop words and / or hashtags
	#meaningful_words = words
	meaningful_words = [w for w in words if not w in stops] 
	#meaningful_words = [w for w in meaningful_words if not w in hashtags]
	#meaningful_words = [w for w in meaningful_words if not w in mentions]   
	#
	# 6. Join the words back into one string separated by space, 
	# and return the result. Or REMOVE THE TARGET 
	#meaningful_words = [w for w in meaningful_words if not w in ['donald', 'trump', 'donaldtrump', 'realdonaldtrump']]   
	return( " ".join( meaningful_words )) 

def clean_tweets(tweets):
	clean_tweets = []
	for i in tweets:
		clean_tweets.append( tweet_to_words( i ) )

	#print(clean_tweets[0]	)
	return clean_tweets

def multi_model_factory():
	names = [
		 "MNB",
		 "SVM",
		 "LR",
		 "RF",
		]

	classifiers = [
		MultinomialNB(),
		LinearSVC(class_weight="balanced"),
		LogisticRegression(class_weight="balanced"),
		RandomForestClassifier(class_weight="balanced"),
		#MLPClassifier()
	]

	parameters = [
			  {'clf__alpha': (1e-2, 1e-3)},
			  {'clf__C': [0.01, 0.1, 1, 10, 100]},
			  {'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__penalty': ['l1', 'l2']},
			  {'clf__max_depth': (1, 10, 50, 100, 200)},
		#	  {'clf__alpha': (1e-2, 1e-3)}
			 ]

	# param_dict = {#'logit__penalty': ['l1', 'l2'] ,
	# 			  'logit__C': [0.01, 100]}

	models = {}
	for name, classifier, params in zip(names, classifiers, parameters):
		ppl = Pipeline([
			('word_ngram', TfidfVectorizer(analyzer = "word",   \
							 tokenizer = None,	\
							 preprocessor = None, \
							 stop_words = set(stopwords.words('english')).discard('not'),   \
							 ngram_range = (1,1))),
			('clf', classifier),
		])
		gs_clf = GridSearchCV(ppl, param_grid=params, n_jobs=-1, cv = 5)
		#clf = gs_clf.fit(X_train, y_train)
		models[name] = gs_clf
	return models





def train_models(training_sets, run):
	models = {}
	model_hyperparameters = {}
	times = {}

	for target in training_sets:
		model_hyperparameters[target] = {}
		times[target] = {}
		#print("Training ", key, ' model...')
		#save training data for bert

		
		
		model_names = multi_model_factory()
		models[target] = {}
		for model in model_names:
			start = time.time()
			model_names[model].fit(clean_tweets(training_sets[target][run]['Tweet'].values), training_sets[target][run]['Stance'].values)
			end = time.time()
			models[target][model] = model_names[model]
			model_hyperparameters[target][model] = model_names[model].best_params_
			times[target][model] = end - start
	return models, model_hyperparameters, times


if __name__ == "__main__":

	test_size = 195 # for all targets

	runs = 5
	training_sets, test_sets = data_load_and_save(datasets = 'all', test_size = test_size, runs = runs) # automatically splits data and saves the splits in ../data/train_test_splits/
	
	for run in range(0, runs):
		
		print(len(training_sets['Donald Trump'][run]))
		print(len(test_sets['Donald Trump'][run]))

	
		models, model_hyperparameters, times = train_models(training_sets, run)
		for model in models:
			for model_type in models[model]:
				# print(model)
				# print(model_type)
				with open('../models/in_domain/%s_%s_%d.pkl' %(model, model_type, run), 'wb') as fid:
					pickle.dump(models[model][model_type], fid) 
				# test_sets[model]['In-domain'] = models[model][model_type].predict(clean_tweets(test_sets[model]['Tweet']))
				# test_sets[model].to_csv("model_outputs/in_domain/%s_%s_%d_%0.3f.csv" %(model, model_type, run, test_size), sep = "\t", index = False)
				# print(len(test_sets[model]))
				# print(classification_report(test_sets[model]['Stance'], test_sets[model]['In-domain']))
				# print()


	print()
	print()
	print()
	for target in ['Hillary Clinton', 'Donald Trump', 'Vladimir Putin', 'Joko Widodo', 
					'Emmanuel Macron', 'Jacob Zuma', 'Recep Tayyip Erdoğan']:
		print(target)
		print(model_hyperparameters[target])
		print(times[target])
		print()
					