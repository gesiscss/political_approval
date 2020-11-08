import pickle
from train_in_domain_models import clean_tweets

def models(tweets, targets, run):
	results = {}
	for n, tweet in enumerate(tweets):
		for model_name in ['LR', 'MNB', 'SVM']:
			if model_name not in results.keys():
				results[model_name] = []
			model = pickle.load(open("../models/in_domain/%s_%s_%d.pkl" %(targets[n], model_name, run), 'rb'))
			results[model_name].append(model.predict(clean_tweets([tweet]))[0])

	return results