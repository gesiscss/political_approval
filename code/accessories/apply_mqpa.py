from __future__ import division
import nltk
import pandas as pd
import os

file = 'accessories/mpqa_dict.csv'
mpqa_csv = pd.read_csv(file, sep = "\t")

mpqa_dict = {}
mpqa_tuples = [(row['word1'], row['priorpolarity']) for index, row in mpqa_csv.iterrows()]

mpqa_dict = dict(mpqa_tuples)


def assign_polarity(message):
	polarities = []
	score = 0.0
	tokens = nltk_tokenize(message.lower())
	for token in tokens:
		if token in mpqa_dict.keys():
			polarities.append(mpqa_dict[token])
	if True:
		score = (1 + float(polarities.count('positive'))) / (1 + float(polarities.count('negative')))
	else:
		score = float(polarities.count('positive'))
	return score, polarities


def nltk_tokenize(message):
	stList = nltk.sent_tokenize(message)
		# word tokenize
	tokens = []
	for sent in stList:
		tokens += nltk.word_tokenize(sent)
	return tokens

if __name__ == "__main__":
	message = "DEMOCRATS WANT OPEN BORDERS. Their actions in the last two years speak to that. They have done nothing on illegal immigration, despite Trump asking them to work with him. They own this. They own the murdered lives of Americans. 2020 will not look kindly on them"
	message = "We hate you president TRUMP."
	print(nltk_tokenize(message.lower())	)
	print(assign_polarity(message))
