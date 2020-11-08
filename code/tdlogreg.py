import sys


from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import re
from nltk.corpus import stopwords # Import the stop word list

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.metrics import classification_report
import pandas as pd
import pickle

class TargetExistenceExtractor(BaseEstimator, TransformerMixin):
	"""Takes in dataframe, extracts road name column, outputs average word length"""

	def __init__(self):
		pass

	def transform(self, raw_tweets, y=None):
		feats = []
		for raw_tweet in raw_tweets:
			tweet_text = raw_tweet 
			#words = tweet_text.lower().split() 
			if 'trump' in tweet_text.lower():
				feats.append(1)
				#print raw_tweet
			else:
				feats.append(0)
			#print(raw_tweet, feats[-1])				
		return pd.DataFrame(feats)

	def fit(self, df, y=None):
		"""Returns `self` unless something different happens in train and test"""
		return self



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

def get_tweet_length(text):
    return len(text)	



if __name__ == "__main__":
	columns = ['Tweet', 'Stance']

	method = sys.argv[1] # 'dssdlogreg' for logreg version of dssd, tdlogreg for logreg version of td-lstm

	if method == 'tdlogreg':
		train = pd.read_csv('../data/train_pytorch_format.tsv', sep = "\t", names = columns)
		test = pd.read_csv('../data/test_pytorch_format.tsv', sep = "\t", names = columns)
	elif method == 'dssdlogreg':
		train = pd.read_csv('../data/trump_autolabelled_pytorch_format.tsv', sep = "\t", names = columns)
		test = pd.read_csv('../data/SemEval2016-Task6-subtaskB-testdata-gold_pytorch_format.tsv', sep = "\t", names = columns)


	# k-fold cross-val grid search training

	classifier = LogisticRegression(class_weight="balanced")
	params = {'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__penalty': ['l1', 'l2', 'none']}


	ppl = Pipeline([
		('feats', FeatureUnion([
				('word_ngram', TfidfVectorizer(analyzer = "word",   \
								 tokenizer = None,	\
								 preprocessor = None, \
								 stop_words = set(stopwords.words('english')).discard('not'),   \
								 ngram_range = (1,2))),
				('target_existence', TargetExistenceExtractor()) # or a transforme
				])),
				('clf', classifier),
			])
	gs_clf = GridSearchCV(ppl, param_grid=params, n_jobs=-1, cv = 5)

	gs_clf.fit(clean_tweets(train['Tweet'].values), train['Stance'].values)
	preds = gs_clf.predict(clean_tweets(test['Tweet']))
	trues = test['Stance']

	print(classification_report(trues, preds))

	with open('../models/%s.pkl' %(method), 'wb') as fid:
		pickle.dump(gs_clf, fid) 