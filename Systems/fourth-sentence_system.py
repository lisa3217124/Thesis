#!/usr/bin/env python3
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_length_sentence(df):
	#returns dataframe with length of fourth sentence and two ending sentences
	length_1 = df['RandomFifthSentenceQuiz1'].str.split().str.len()
	length_2 = df['RandomFifthSentenceQuiz2'].str.split().str.len()
	length_4 = df['InputSentence4'].str.split().str.len()
	df_length = pd.DataFrame({'length_4': list(length_4)+list(length_4), 'length_end': list(length_1)+list(length_2)})
	return df_length

def get_all_sentences(df):
	#returns list with removed ending marks and commas and lowercase fourth sentence and two ending sentences
	sent_1 = df['RandomFifthSentenceQuiz1'].str.strip('.?!').str.lower().str.replace(',', '') 
	sent_2 = df['RandomFifthSentenceQuiz2'].str.strip('.?!').str.lower().str.replace(',', '') 
	sent_4 = df['InputSentence4'].str.strip('.?!').str.lower().str.replace(',', '') 
	sentences = list(sent_1)+list(sent_2)+list(sent_4)
	return sentences

def find_tagwords(sentences):
	#returns the tagwords
	tfidf_vectorizer = TfidfVectorizer()
	tfidf_scores = tfidf_vectorizer.fit_transform(sentences)
	tfidf = pd.DataFrame(tfidf_scores.toarray(), columns=tfidf_vectorizer.get_feature_names())
	tfidf.loc['sum'] = tfidf.sum(axis = 0)
	tfidf = tfidf.sort_values(by='sum', axis=1, ascending = False) 
	tagwords = list(tfidf)[0:100]
	return tagwords

def replace_tagwords_with_postag(sentences, tagwords):
	#returns list with fourth and two ending sentences in which the tagwords are replaced with their part-of-speech tag
	new_sentences = []
	for sentence in sentences:
		new_sentence = []
		sent = sentence.split()
		for word in sent: # we first POS tag the whole sentence to get better POS tags for ambigious words, like work & to work : NN  & VB
			if word in tagwords:
				pos_sent = dict(nltk.pos_tag(sent))
				pos = pos_sent[word]
				new_sentence.append(pos)
			else:
				new_sentence.append(word)
		new_sentences.append(' '.join(new_sentence))
	return new_sentences

def combine_first_end(sentences):
	#returns list of concatenated fourth sentence with first ending sentence and concatenated fourth sentence with second ending sentence
	sent_1 = sentences[:int(len(sentences)/3)]
	sent_2 = sentences[int(len(sentences)/3):int(2*(len(sentences)/3))]
	sent_4 = sentences[int(2*(len(sentences)/3)):]
	comb_sent4_end1 = [str(sent_4[i])+' '+sent_1[i] for i in range(len(sent_1))]
	comb_sent4_end2 = [str(sent_4[i])+' '+sent_2[i] for i in range(len(sent_2))]
	return comb_sent4_end1 + comb_sent4_end2

def get_word_ngrams(sentences):
	#returns dataframe of unigrams, bigrams and trigrams as features and their counts
	count_vectorizer = CountVectorizer(ngram_range=(1,3), lowercase = False) # no lowercase bacause of IN would be changed to in
	ngrams = count_vectorizer.fit_transform(sentences) 
	word_features = pd.DataFrame(ngrams.toarray(), columns=count_vectorizer.get_feature_names())
	return word_features


def get_char_quadrigrams(sentences):
	#returns dataframe of character quadrigrams as features and their counts
	char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(4,4), lowercase = False) # no lowercase bacause of IN would be changed to in
	quadrigrams = char_vectorizer.fit_transform(sentences) 
	char_features = pd.DataFrame(quadrigrams.toarray(), columns=char_vectorizer.get_feature_names())
	return char_features

def get_sentiment(df):
	#returns dataframe with sentiment scores for each of the two ending sentences and fourth sentence
	analyser = SentimentIntensityAnalyzer()	
	end1 = list(df['RandomFifthSentenceQuiz1'])
	end2 = list(df['RandomFifthSentenceQuiz2'])
	sent4 = list(df['InputSentence4'])
	pos, neg, neu, comp = [], [], [], []
	pos4, neg4, neu4, comp4 = [], [], [], []

	for sent in end1:
		sentiment = analyser.polarity_scores(sent)
		pos.append(sentiment['pos'])
		neg.append(sentiment['neg'])
		neu.append(sentiment['neu'])
		comp.append(sentiment['compound'])
	for sent in end2:
		sentiment = analyser.polarity_scores(sent)
		pos.append(sentiment['pos'])
		neg.append(sentiment['neg'])
		neu.append(sentiment['neu'])
		comp.append(sentiment['compound'])
	for sent in sent4:
		sentiment = analyser.polarity_scores(sent)
		pos4.append(sentiment['pos'])
		neg4.append(sentiment['neg'])
		neu4.append(sentiment['neu'])
		comp4.append(sentiment['compound'])

	pos4, neg4, neu4, comp4 = pos4 + pos4, neg4 + neg4, neu4 + neu4, comp4 + comp4
	df_sentiment = pd.DataFrame({'pos4': pos4, 'neg4': neg4, 'neu4': neu4, 'comp4': comp4, 'pos': pos, 'neg': neg, 'neu': neu, 'comp': comp})
	return df_sentiment

def answers(right_endings):
	#adds column with for right ending 1 and wrong ending 0
	#first all first ending sentences and then all second ending sentences
	first, second = right_endings, right_endings
	first = first.replace(2, 0)
	second = second.replace(1, 0)
	second = second.replace(2, 1)
	answers = pd.concat([first, second], axis = 0, sort = False, ignore_index = True)
	return answers

def get_word_ngrams_test(sentences_train, sentences):
	#returns dataframe of unigrams, bigrams and trigrams as features from training set and the counts of the development/test set
	count_vectorizer = CountVectorizer(ngram_range=(1,3), lowercase = False) # no lowercase bacause of IN would be changed to in 
	ngrams = count_vectorizer.fit(sentences_train)
	ngrams = count_vectorizer.transform(sentences)
	word_features = pd.DataFrame(ngrams.toarray(), columns=count_vectorizer.get_feature_names())
	return word_features

def get_char_quadrigrams_test(sentences_train, sentences):
	#returns dataframe of character quadrigrams as features from training set and the counts of the development/test set
	char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(4,4), lowercase = False) # no lowercase bacause of IN would be changed to in
	quadrigrams = char_vectorizer.fit(sentences_train) 
	quadrigrams = char_vectorizer.transform(sentences) 
	char_features = pd.DataFrame(quadrigrams.toarray(), columns=char_vectorizer.get_feature_names())
	return char_features

def get_performance(X_train, X_test, y_train, y_test, df_dev):
	#predicts for each ending sentence if it is the right or wrong ending and also predicts the probability of being right and wrong ending
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		logmodel = LogisticRegression(solver='lbfgs')
		logmodel.fit(X_train,y_train)
		predictions = logmodel.predict(X_test)
		prob_predictions = logmodel.predict_proba(X_test)
	first_end, second_end = predictions[:int(len(predictions)/2)], predictions[int(len(predictions)/2):]
	first_prob, second_prob = prob_predictions[:int(len(prob_predictions)/2)], prob_predictions[int(len(prob_predictions)/2):]
	first_prob0, first_prob1, second_prob0, second_prob1 = [], [], [], []
	for prob in first_prob:
		first_prob0.append(prob[0])
		first_prob1.append(prob[1])
	for prob in second_prob:
		second_prob0.append(prob[0])
		second_prob1.append(prob[1])

	#if both endings get same prediction, ending with highest probability of prediction gets that prediction and other ending gets opposite prediction
	df = pd.DataFrame({'first_end':first_end, 'second_end': second_end, 'first_prob0': first_prob0, 'first_prob1': first_prob1, 'second_prob0': second_prob0, 'second_prob1': second_prob1})
	answers = []
	for index, row in df.iterrows():
		if row.iloc[0] == row.iloc[1] and row.iloc[0]==0:
			if row.iloc[2] > row.iloc[4]:
				row.iloc[1]=1
			else:
				row.iloc[0] = 1
		elif row.iloc[0] == row.iloc[1] and row.iloc[0]==1:
			if row.iloc[3] > row.iloc[5]:
				row.iloc[1]=0
			else:
				row.iloc[0]=0
		#transforms predictions to the number of the right ending sentence of each story, thus 1 if first ending is right ending, 2 if second ending is right ending
		if row.iloc[0]==1:
			answers.append(1)
		else:
			answers.append(2)
	modified_predictions = answers
	storyid = df_dev.loc[:,'InputStoryid']
	new_y_test = df_dev.loc[:,'AnswerRightEnding']

	#prints story id with predicted right ending
	for i in range(len(storyid)):
		print(str(storyid[i])+ " " +str(modified_predictions[i]))

	#evaluation data
	# print(metrics.accuracy_score(new_y_test, modified_predictions), '\t', 
	# 	metrics.precision_score(new_y_test, modified_predictions), '\t',
	# 	metrics.recall_score(new_y_test, modified_predictions), '\t',
	# 	metrics.f1_score(new_y_test, modified_predictions), '\t')


def main():
	#TRAIN SET
	df_train = pd.read_csv('../Data/train2016.csv', delimiter=",") # use '../Data/train2018.csv' for 2018 train set
	data_train = df_train.iloc[:,[0,4,5,6,7]]
	all_sentences_train = get_all_sentences(data_train)
	tagwords_train = find_tagwords(all_sentences_train)
	sentences_postag_train = replace_tagwords_with_postag(all_sentences_train, tagwords_train)
	combined_sentences_train = combine_first_end(sentences_postag_train)

	#features
	length_train = get_length_sentence(data_train)
	word_train = get_word_ngrams(combined_sentences_train)
	char_train = get_char_quadrigrams(combined_sentences_train)
	sentiment_train = get_sentiment(data_train)
	features_train = pd.concat([sentiment_train, length_train, word_train, char_train], axis = 1, sort=False)
	features_train = features_train.apply(pd.to_numeric, downcast='integer')

	# # if you only want to get performance of a few features, comment out other features and modify features_train, for example:
	# length_train = get_length_sentence(data_train)	
	# # word_train = get_word_ngrams(combined_sentences_train)
	# # char_train = get_char_quadrigrams(combined_sentences_train)
	# sentiment_train = get_sentiment(data_train)
	# features_train = pd.concat([sentiment_train, length_train], axis = 1, sort=False)


	answers_train = answers(data_train.iloc[:,4])

	#DEVELOPMENT SET
	# df_dev = pd.read_csv('../Data/dev2016.csv', delimiter=",") # use '../Data/dev2018.csv' for 2018 development set
	# data_dev = df_dev.iloc[:,[0,4,5,6,7]]

	# all_sentences_dev = get_all_sentences(data_dev)
	# sentences_postag_dev = replace_tagwords_with_postag(all_sentences_dev, tagwords_train) 
	# combined_sentences_dev = combine_first_end(sentences_postag_dev)

	# #features
	# length_dev = get_length_sentence(data_dev)
	# word_dev = get_word_ngrams_test(combined_sentences_train, combined_sentences_dev)
	# char_dev = get_char_quadrigrams_test(combined_sentences_train, combined_sentences_dev)
	# sentiment_dev = get_sentiment(data_dev)
	# features_dev = pd.concat([sentiment_dev, length_dev, word_dev, char_dev], axis = 1, sort=False)
	# features_dev = features_dev.apply(pd.to_numeric, downcast='integer')

	# # #if you only want to get performance of a few feature, comment out other features and modify features_dev, for example:
	# # length_dev = get_length_sentence(data_dev)
	# # # word_dev = get_word_ngrams_test(combined_sentences_train, combined_sentences_dev)
	# # # char_dev = get_char_quadrigrams_test(combined_sentences_train, combined_sentences_dev)
	# # sentiment_dev = get_sentiment(data_dev)
	# # features_dev = pd.concat([sentiment_dev, length_dev], axis = 1, sort=False)


	# answers_dev = answers(data_dev.iloc[:,4]) 
	# get_performance(features_train, features_dev, answers_train, answers_dev, df_dev)


	#TEST SET
	df_test = pd.read_csv('../Data/cloze_test_test__spring2016 - cloze_test_ALL_test.csv', delimiter=",") # use "../Data/cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv" for 2018 development set
	data_test = df_test.iloc[:,[0,4,5,6,7]]

	all_sentences_test = get_all_sentences(data_test)
	sentences_postag_test = replace_tagwords_with_postag(all_sentences_test, tagwords_train) 
	combined_sentences_test = combine_first_end(sentences_postag_test)

	#features
	length_test = get_length_sentence(data_test)
	word_test = get_word_ngrams_test(combined_sentences_train, combined_sentences_test)
	char_test = get_char_quadrigrams_test(combined_sentences_train, combined_sentences_test)
	sentiment_test = get_sentiment(data_test)
	features_test = pd.concat([sentiment_test, length_test, word_test, char_test], axis = 1, sort=False)
	features_test = features_test.apply(pd.to_numeric, downcast='integer')

	# #if you only want to get performance of a few feature, comment out other features and modify features_test, for example:
	# length_test = get_length_sentence(data_test)
	# # word_test = get_word_ngrams_test(combined_sentences_train, combined_sentences_test)
	# # char_test = get_char_quadrigrams_test(combined_sentences_train, combined_sentences_test)
	# sentiment_test = get_sentiment(data_test)
	# features_test = pd.concat([sentiment_test, length_test], axis = 1, sort=False)

	answers_test = answers(data_test.iloc[:,4]) 
	get_performance(features_train, features_test, answers_train, answers_test, df_test)


if __name__ == '__main__':
	main()
