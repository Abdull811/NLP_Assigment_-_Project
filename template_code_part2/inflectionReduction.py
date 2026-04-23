from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, WordNetLemmatizer, PorterStemmer, etc.)
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download wordnet data for lemmatization
nltk.download('wordnet') 
nltk.download("omw-1.4") # Download open multilingual wordnet data for lemmatization

class InflectionReduction:
	def __init__(self):
		# Prepare the Porter stemmer and WordNet lemmatizer
		self.porter = PorterStemmer()
		self.lemmatizer = WordNetLemmatizer()
		 
	def porterStemmer(self, text):
		"""
		Inflection Reduction using Porter Stemmer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed tokens representing a sentence
		"""

		reducedText = None

		# Fill in code here
		reducedText = [] # empty list to store the stemmed output for all sentences.
		# Function that iterate through each sentene 
		for sentence in text:
			reducedSentence = []
			# Function that iterate through each token in the sentence
			for token in sentence:
				# Convert the token to its stemmed form using Porter Stemmer
				# It reduce the token to its root form by removing suffixes
				stemsentence = self.porter.stem(token)
				reducedSentence.append(stemsentence) # Save the stemmed sentence
			reducedText.append(reducedSentence)	

		return reducedText


	def wordnetLemmatizer(self, text):
		"""
		Inflection Reduction using WordNet Lemmatizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""

		reducedText = None

		# Fill in code here
		# Store the lemmatized output for all sentences.
		reducedText = []
		for sentence in text:
			reducedSentence = []
			for token in sentence:
				# Convert each token to its base dictionary form.
				lemsentence = self.lemmatizer.lemmatize(token)
				reducedSentence.append(lemsentence)
			reducedText.append(reducedSentence)

		return reducedText


	def reduce(self, text):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""

		reducedText = None

		# Fill in code here
		# Use Porter stemming as the default method.
		# reducedText = self.porterStemmer(text)  # For Porter sstemmer
		reducedText = self.wordnetLemmatizer(text) # For WordNet Lemmatizer

		return reducedText
