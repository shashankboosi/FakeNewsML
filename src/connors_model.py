"""
COMP9417
Assignment
Author: Connor McLeod (z5058240)
connors_model.py: Main calls on this file for model implementation

 	To do:
	- learn from training subset using several learning frameworks (random forest)
	- apply models to validation subset
	- assess results of models on validation set

"""

import sys, csv, string, re, pprint, math, operator
import pandas as pd
from sklearn.model_selection import train_test_split
from output.scorer_modified import *
from sklearn.ensemble import RandomForestClassifier

ignore_words = ['in','the','to','a','of','and','that','is','was','on','for','he','him','it','his','have','as','has','be','an','are','this','had','us','at','by','her','she','or','i']

def write_to_csv(filename, heading, content):
	"""
	Function to write desired output to csv file
	:param filename: the name of the file
	:param heading: a Python list of the headers for the csv ['Headline','Body ID','Stance']
	:param content: a Python list of lists where each list has the same structure as the header
	:return: created csv filepath
	"""
	with open(filename, mode='w') as file:
		csv_writer = csv.writer(file)
		csv_writer.writerow(heading)
		for row in content:
			csv_writer.writerow(row)
	print (filename, " created")
	return filename



def read_articles(df):
	"""
	Function to analyse the articles of the training set
	:param df: Pandas dataframe containing training sample
	:return wordcount: Dictionary containing stance, headline word count and body word count for each article (index on Body ID)
	"""


	words = {}
	""" 
	{
		'Body_ID':
		{
			'Headline_og': ['Headline text...'],
			'Headline': ['word1', 'word2', ...],
			'Stance': ['unrelated','discuss','agree','disagree'],
			'articleBody': ['word1', 'word2', ...]
		}
	}
	"""

	wordcount = {}
	"""
	{
		'Body_ID':
		{
			'Headline_og': ['Headline text...'],
			'Headline': {'word1': #, 'word2': #, ...},
			'Stance': ['unrelated','discuss','agree','disagree'],
			'articleBody': {'word1': #, 'word2': #, ...}
		}
	}
	"""

	stance_count = {}
	stance_count['unrelated'] = {}
	stance_count['discuss'] = {}
	stance_count['agree'] = {}
	stance_count['disagree'] = {}
	"""
	{
		'Stance': {'word1': #, 'word2': #, ...}
	}
	"""

	# for each article
	for index, article in df.iterrows():

		# prepare dictionary for article entry
		Body_ID = article['Body ID']
		words[Body_ID] = {}
		words[Body_ID]['Headline_og'] = article['Headline']
		wordcount[Body_ID] = {}
		wordcount[Body_ID]['Headline_og'] = article['Headline']

		# for headline and body of article
		for text in ['Headline', 'articleBody']:
			# sanitise article text
			article[text] = article[text].lower()
			article[text] = article[text].translate(str.maketrans('','',string.punctuation))
			article[text] = article[text].replace('\n',' ').replace('\r',' ')
			article[text] = article[text].replace('“','')
			article[text] = article[text].replace('‘','')
			article[text] = re.sub(' +',' ',article[text])

			# separate article words
			article_words = (article[text].split(' '))
			article_words = [word for word in article_words if word not in ignore_words]
			words[Body_ID][text] = article_words

			# generate article word count
			wordcount[Body_ID][text] = {}
			wordcount[Body_ID][text] = {word:article_words.count(word) for word in article_words}

			# store word count in stance_count dictionary
			stance = article['Stance']
			for word in article_words:
				if (word not in stance_count[stance]):
					stance_count[stance][word] = 1
				else:
					stance_count[stance][word] += 1
		
		# store stance into article's dictionary entry
		words[Body_ID]['Stance'] = article['Stance']
		wordcount[Body_ID]['Stance'] = article['Stance']

		# pprint.pprint(stance_count) # pprint.pprint(words)
	return wordcount, stance_count



def stance_prediction(wordcount, stance_count):
	"""
	Function to predict article stance given the article headline and word count
	:param wordcount_dict: dictionary containing article's wordcount
	:param stance_count: xxx
	:return prediction: list of lists containing 'Headline', Body ID', 'Stance' where stance is predicted
	"""

	prediction = {}
	for key, val in wordcount.items():
		prediction[key] = {}
		P_weight = {}
		for word in val['articleBody']:
			for stance in ['unrelated', 'discuss', 'agree', 'disagree']:
				P_weight[stance] = 0
				probability = stance_count[stance].get(word)
				stance_total = sum(stance_count[stance].values())

				if (probability):
					# print (word, probability, stance_total)
					probability = math.log(probability/stance_total)
					P_weight[stance] += probability
		
		prediction[key]['Headline'] = val['Headline_og']	
		prediction[key]['Stance'] = max(P_weight.items(), key=operator.itemgetter(1))[0]

	# pprint.pprint (prediction)
	return prediction


def check_predictions(predictions, wordcount):
	"""
	Function description
	:params:
	:return:
	"""

	# comment xxx
	heading = ['Headline', 'Body ID', 'Stance']

	# comment xxx
	test_set = []
	for key, val in predictions.items():
		test_set.append([val['Headline'],key,val['Stance']])
	test_csv = write_to_csv('output/cm_test_out.csv', heading, test_set)

	# comment xxx
	true_set = []
	for key, val in predictions.items():
		true_headline = wordcount[key]['Headline_og']
		true_stance = wordcount[key]['Stance']
		true_set.append([true_headline, key, true_stance])
	true_csv = write_to_csv('output/cm_true_out.csv', heading, true_set)

	# comment xxx
	print("\nCM Naive-Bayes Results:\n")
	report_score(true_csv, test_csv)


def connors_model():

	df_bodies = pd.read_csv("data/train_bodies.csv")
	df_stances = pd.read_csv("data/train_stances.csv")

	df = pd.merge(df_bodies, df_stances, on="Body ID") # df.columns.values: ['Body ID' 'articleBody' 'Headline' 'Stance']
	# df.set_index('Body ID', inplace=True)
	train_df, validate_df = train_test_split(df, test_size=0.9995, random_state=0)
	
	# comment xxx
	wordcount, stance_count = read_articles(train_df)

	# comment xxx
	predictions = stance_prediction(wordcount, stance_count)

	# comment xxx
	check_predictions(predictions, wordcount)




