"""
COMP9417
Assignment
Author: Connor McLeod (z5058240)
connors_model.py: Main calls on this file for model implementation
"""

import sys, csv
import pandas as pd
from sklearn.model_selection import train_test_split

def write_to_csv(filename, heading, content):
	"""
	Function to write desired output to csv file
	:param filename: the name of the file
	:param heading: a Python list of the headers for the csv
	:param content: a Python list of lists containing [headline, body ID, stance (prediction)]
	:return: None
	"""
	with open(filename, mode='w') as test_file:
		csv_writer = csv.writer(test_file)
		csv_writer.writerow(heading)
		for row in content:
			csv_writer.writerow(row)
	print (filename, " created")


def connors_model():

	#df_bodies = pd.read_csv("data/train_bodies.csv")
	#df_stances = pd.read_csv("data/train_stances.csv")

	# df = pd.merge(df_bodies, df_stances, on="Body ID")
	# print column headings
	# print ("Dataframe column headings:\n", df.columns.values)

	# data_x = df[["articleBody","Headline"]].values
	# data_y = df["Stance"].values
	# train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.20, random_state=0)

	# to do:
	# learn from training subset using several learning frameworks (k NN, decision tree, etc)
	# apply models to validation subset
	# assess results of models on validation set

	heading = ['Headline', 'Body ID', 'Stance']
	true_set = [
		['Ferguson riots: Pregnant woman loses eye after cops fire BEAN BAG round through car window','2008','unrelated'],
		['Crazy Conservatives Are Sure a Gitmo Detainee Killed James Foley','1550','unrelated'],
		['A Russian Guy Says His Justin Bieber Ringtone Saved Him From A Bear Attack','2','unrelated']
	]
	test_set = [
		['Ferguson riots: Pregnant woman loses eye after cops fire BEAN BAG round through car window','2008','discuss'],
		['Crazy Conservatives Are Sure a Gitmo Detainee Killed James Foley','1550','unrelated'],
		['A Russian Guy Says His Justin Bieber Ringtone Saved Him From A Bear Attack','2','unrelated']
	]

	write_to_csv('output/cm_true_out.csv', heading, true_set)
	write_to_csv('output/cm_test_out.csv', heading, test_set)