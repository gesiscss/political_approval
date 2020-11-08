import pickle
from tdlogreg import TargetExistenceExtractor

def logreg(tweets, modeltype = 'tdlogreg'):
	mapping = {0 : "NONE", 1 : "FAVOR", 2 : "AGAINST"}
	model = pickle.load(open("../models/%s.pkl" %(modeltype), 'rb'))
	return [mapping[i] for i in model.predict(tweets)]

def td_lstm(tweets):
	from os import listdir
	from os.path import isfile, join
	targets = ['Hillary', 'Donald', 'Emmanuel', 'Jacob', 'Recep', 'Joko', 'Vladimir']
	all_runs = []
	runs = 5

	for i in range(0, runs):
		all_dfs = pd.DataFrame()
		for target in targets:
			with open("model_outputs/TD-LSTM/%s_run_%d.txt" %(target, i)) as f:
				content = f.readlines()
			content = [x.strip() for x in content] 

			df = pd.read_csv("TD-LSTM/data/twitter/%s_tdlstm_format_anchor.csv" %(target), sep = "\t")
			df['Predicted_Stance'] = content
			all_dfs = all_dfs.append(df)

	
		all_data = pd.read_csv("TD-LSTM/data/twitter/all_data_tdlstm_format_anchor.csv", sep = '\t')

		all_dfs = all_dfs[['ID', 'Target', 'Tweet', 'Predicted_Stance']]
		all_dfs = all_data.merge(all_dfs, on = ['ID', 'Tweet', 'Target'])


		all_runs.append(all_dfs['Predicted_Stance'].values)

	return all_runs





def dssd_old(tweets):
	from os import listdir
	from os.path import isfile, join
	targets = ['Hillary', 'Donald', 'Emmanuel', 'Jacob', 'Recep', 'Joko', 'Vladimir']

	all_runs = []
	for run in range(0, 5):
		all_dfs = pd.DataFrame()
		for target in targets:
			mypath = "model_outputs/DSSD/run_%s/run_%d/" %(target, run)
			onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) if 'txt' in f]
			df = pd.read_csv(mypath+onlyfiles[1], sep = "\t")
			all_dfs = all_dfs.append(df)

	
		all_data = pd.read_csv("DSSD/data/all_test_data_dssd_format.csv", sep = '\t')
		print(len(all_data))

		all_dfs.columns = ['ID', 'Target', 'Tweet', 'Predicted_Stance']
		all_dfs = all_data.merge(all_dfs, on = ['ID', 'Tweet', 'Target'])
		all_runs.append(all_dfs)
	print(len(all_dfs['Predicted_Stance'].values))
	return all_runs



# ------------------------------------------------------------------------------------------------------
# Trained methods	


def stance_detection_sm(tweets, run):
	import pickle 
	with open('models/stance_detection_sm/run_balanced_%d.pkl' %run, 'rb') as f:
		model = pickle.load(f)
	return model.predict(clean_tweets(tweets))
