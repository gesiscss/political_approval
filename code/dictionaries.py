def vader(tweets):
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	analyzer = SentimentIntensityAnalyzer()
	tweets_polarity = []
	y_score = []

	for sentence in tweets:
		vader_score = analyzer.polarity_scores(sentence)
		tweets_polarity.append(vader_score['compound'])
		del vader_score['compound']
		y_score.append(vader_score.values())

	tweets_polarity = ['AGAINST' if i < -0.1 else 'FAVOR' if i > 0.1 else 'NONE' for i in tweets_polarity]
	return tweets_polarity


def labmt(tweets):
	from accessories.labmt import score
	tweets_polarity = []
	for sentence in tweets:
		tweets_polarity.append(score(sentence))

	tweets_polarity = ['AGAINST' if i < -0.1 else 'FAVOR' if i > 0.1 else 'NONE' for i in tweets_polarity]
	# get out of accessories 
	return tweets_polarity

def mpqa(tweets):
	from accessories.apply_mqpa import assign_polarity	
	tweets_polarity = []
	for sentence in tweets:
		tweets_polarity.append(assign_polarity(sentence)[0])

	tweets_polarity = ['AGAINST' if i < 0.9 else 'FAVOR' if i > 1.1 else 'NONE' for i in tweets_polarity]
	return tweets_polarity


def stb(tweets):
	# only run once and save because it takes some time

	# tweets = data['Tweet']
	# import os
	# from sentitreebank.sentitreebank import tag_sent
	# tweets_polarity = []
	# for sentence in tweets:
	# 	try:
	# 		tweets_polarity.append(tag_sent(sentence.replace('"', '').replace("'", '')))
	# 	except:
	# 		print(sentence)
	# 		tweets_polarity.append('NONE')
	# 		pass
	# tweets_polarity = ['AGAINST' if i in [b'Negative', b'Very Negative'] else 'FAVOR' if i in [b'Positive', b'Very Positive'] else 'NONE' for i in tweets_polarity]
	# # get out of sentitreebank 
	# os.chdir("../")
	# # print(os.getcwd())

	# data['SentiTreeBank'] = tweets_polarity
	# data.to_csv('model_outputs/STB/run_0.csv', sep = '\t', index = False)
	# with open('model_outputs/STB/run_0.txt', 'w') as f:
	# 	for i in tweets_polarity:
	# 		f.write("%s\n" %i)

	data = pd.read_csv('model_outputs/STB/run_0.csv', sep = '\t')
	return data
	# with open("model_outputs/STB/run_0.txt") as f:
	# 		content = f.readlines()
	# content = [x.strip() for x in content] 
	# print(len(content))
	# return content


# sentistrength
def sts(tweets):
	from accessories.sentis.sentistrength_calc import rate_sentiment
	tweets_polarity = []
	for n, tweet in enumerate(tweets):
		if n % 1000 == 0:
			print(n, " done.")
		#try:
		tweets_polarity.append(rate_sentiment(tweet.replace("\n", ' ')))
		#except:
		#	print(tweet)
		#	pass
	tweets_polarity = ['AGAINST' if i == -1 else 'FAVOR' if i == 1 else 'NONE' for i in tweets_polarity]
	

	return tweets_polarity


