# library
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# function to turn review into words list
def review_to_wordslist (raw_review):
	# remove tags and markup
	review_text = BeautifulSoup(raw_review, 'lxml').get_text()

	# remove anything not letters
	letter_only = re.sub("[^a-zA-Z]", " ", review_text)

	# convert to lower case and split into words
	words = letter_only.lower().split()

	# convert stop words into a set
	stop_words = set(stopwords.words("english")) # searching faster if convert a list to a set first

	# remove stop words
	fix_words = [w for w in words if not w in stop_words]

	return(" ".join(fix_words))


if __name__ == '__main__':
	# read training dataset
	train = pd.read_csv("./labeledTrainData.tsv", header = 0, delimiter = "\t", quoting=3)
	test = pd.read_csv("./testData.tsv", header = 0, delimiter = "\t", quoting=3)

	# print training dataset size and 1st review
	print 'Data set shape'
	print train.shape
	##print '\n'
	##print '1st review in dataset'
	##print train.review[0]

	# print review after clean up
	##print '\n'
	##print 'review after clean up'
	##print review_to_wordslist(train.review[0])

	# Initialize empty list to hold clean reviews
	clean_train_reviews = []

	# loop over each review in the dataset
	print '\n'
	print 'Cleaning reviews ...'
	for i in xrange(0, len(train['review'])):
		# show the progress of cleaning
		if((i+1) % 1000 ==0):
			print 'review %d of %d' % (i+1, len(train['review']))
		clean_train_reviews.append(review_to_wordslist(train['review'][i]))

#	print '\n 1st clean review'
#	print clean_train_reviews[0]

	# Creating deatures from a bag of words (scikit-learn)
	# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
	vectorizer = CountVectorizer(analyzer = "word", \
								tokenizer = None,   \
								preprocessor = None,\
								stop_words = None,  \
								max_features = 5000) # each features will be a column in the result

	train_data_features = vectorizer.fit_transform(clean_train_reviews)

	# Numpy arrays are easy to work with, so convert the result to an array
	train_data_features = train_data_features.toarray()

	# get features name then counts of each word/features
#	features_names = vectorizer.get_feature_names()
#	dist = np.sum(train_data_features, axis = 0)
#	for tag, count in zip(features_names, dist):
#		print tag, count

	# Build random forest model
	model = RandomForestClassifier(n_estimators = 100)
	model.fit(train_data_features, train['sentiment'])

	# Initialize empty list to hold clean reviews
	clean_test_reviews = []

	# clean up the new review
	print '\n'
	print 'Cleaning reviews ...'
	for i in xrange(0, len(test['review'])):
		# show the progress of cleaning
		if((i+1) % 1000 ==0):
			print 'review %d of %d' % (i+1, len(test['review']))
		clean_test_reviews.append(review_to_wordslist(test['review'][i]))
	
	# get bag of words for test data, convert it into numpy array
	test_data_features = vectorizer.fit_transform(clean_test_reviews)
	test_data_features = test_data_features.toarray()

	# prediction using random forest model
	print "predicting results for test data"
	result = model.predict(test_data_features)

	# get output of the results
	output = pd.DataFrame(data={"contant":test["review"], "sentiment":result})

	# write output file
	output.to_csv("./bag_of_word.csv", index=False)
