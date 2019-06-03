#!/usr/bin/env python3
import pandas as pd 
from sklearn.model_selection import train_test_split

def read_data_2016():
	#reads file and returns it into dataframe format
	df = pd.read_csv("Data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv", delimiter=",")
	df =df.set_index('InputStoryid')
	return df


def split_train_test_2016(data):
	#splits dataframe into train and development data
	train, test = train_test_split(data, test_size=0.1)
	train2016 = train.to_csv('Data/train2016.csv')
	test2016 = test.to_csv('Data/dev2016.csv')

def read_data_2018():
	#reads file and returns it into dataframe format
	df = pd.read_csv("Data/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv", delimiter=",")
	df =df.set_index('InputStoryid')
	return df


def split_train_test_2018(data):
	#splits dataframe into train and development data
	train, test = train_test_split(data, test_size=0.1)
	train2018 = train.to_csv('Data/train2018.csv')
	test2018 = test.to_csv('Data/dev2018.csv')


def main():
	data2016 = read_data_2016()
	split_train_test_2016(data2016)
	data2018 = read_data_2018()
	split_train_test_2018(data2018)


#NOTES
#modified by hand test final 2018; add columnames AnswerRightEnding and the answers
# 27490bbd-432e-40da-9731-779a256e7d16 had last sentence split up into two columns so had to join them: row 111


if __name__ == '__main__':
	main()