"""
COMP9417
Assignment
main.c: Main file for program execution
"""

import pandas as pd


if __name__ == "__main__":

	df_bodies = pd.read_csv("fakenews_dataset/train_bodies.csv")
	df_stances = pd.read_csv("fakenews_dataset/train_stances.csv")

	# to do:
	# separate df_bodies and df_stances into train/validate/test
	# learn from training subset using several learning frameworks (k NN, decision tree, etc)
	# apply models to validation subset
	# assess results of models on validation set


	# admin to do:
	# create virtual environment and requirements.txt file