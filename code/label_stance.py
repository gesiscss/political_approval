import pandas as pd
from dictionaries import vader, labmt, mpqa, sts
from pretrained_models import logreg, stance_detection_sm, dssd_old
from in_domain import models
from tdlogreg import TargetExistenceExtractor
from util import directed


def run_methods():

	runs = 5
	for run in range(0, runs):
		data = pd.read_csv("../data/train_test_split/all_test_%d.csv" %(run), sep = "\t")
		data['Direct_Stance'] = [directed(row['Tweet'], row['Target']) for n, row in data.iterrows()]

		print("\n\nVADER\n\n")
		data['VADER_%d' %run] = vader(data['Tweet'].values)

		print("\n\nLabMT\n\n")
		data['LabMT_%d' %run] = labmt(data['Tweet'].values)

		print("\n\nMPQA\n\n")
		data['MPQA_%d' %run] = mpqa(data['Tweet'].values)

		# print("\n\nSTS\n\n")
		# data['STS_%d' %run] = sts(data['Tweet'].values)

		print("\n\nTD-LR\n\n")
		data['TD-LR_%d' %run] = logreg(data['Tweet'].values, modeltype = 'tdlogreg')


		print("\n\nDSSD\n\n")
		data['DSSD_%d' %run] = dssd_old(run = run)

		print("\n\nall in-domain\n\n")
		in_domain_results = models(data['Tweet'].values, data['Target'].values, run)
		data['in_domain_LR_%d' %run] = in_domain_results['LR']
		data['in_domain_MNB_%d' %run] = in_domain_results['MNB']
		data['in_domain_SVM_%d' %run] = in_domain_results['SVM']
	
		# print(data.head())
		# print(len(data))
		# Drop tweet column before saving as csv
		data = data.drop('Tweet', 1)
		data.to_csv("../outputs/all_test_%d_results.csv" %(run), sep = "\t")
	



if __name__ == "__main__":
	# to generate test data in the format of these methods, already generated
	# data_save(format = 'tdlstm', whole = True)
	# data_save(format = 'dssd')
	
	run_methods()