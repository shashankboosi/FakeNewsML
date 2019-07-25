"""
COMP9417
Assignment
main.py: Main file for program execution
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


if __name__ == "__main__":
    df_bodies = pd.read_csv("data/train_bodies.csv")
    df_stances = pd.read_csv("data/train_stances.csv")

	df = pd.merge(df_bodies, df_stances, on="Body ID")

	# print column headings
	print ("Dataframe column headings:\n", df.columns.values)

	data_x = df[["articleBody","Headline"]].values
	data_y = df["Stance"].values
	train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.20, random_state=0)



	# to do:
	# learn from training subset using several learning frameworks (k NN, decision tree, etc)
	# apply models to validation subset
	# assess results of models on validation set