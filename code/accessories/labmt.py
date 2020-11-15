

import pandas as pd
 
url = 'http://www.plosone.org/article/fetchSingleRepresentation.action?uri=info:doi/10.1371/journal.pone.0026752.s001'
labmt = pd.read_csv(url, skiprows=2, sep='\t', index_col=0)
 
average = labmt.happiness_average.mean()
happiness = (labmt.happiness_average - average).to_dict()
 
def score(text):
    words = text.split()
    return sum([happiness.get(word.lower(), 0.0) for word in words]) / len(words)

if __name__ == "__main__":

	print(score('Indira is Incredible and excellent')    )
	print(score('Trump is Incredible and excellent')    )
	print(score('Obama is Incredible and excellent')    )