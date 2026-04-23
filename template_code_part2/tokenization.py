from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)
import re
import spacy
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():

	def __init__(self):
		
		# Prepare the Penn Treebank tokenizer 
		self.ptb_tokenizer = TreebankWordTokenizer()
		# Load spacy model
		self.nlp = spacy.load("en_core_web_sm")
		
	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		# Fill in code here
		# Save tokens for each sentence 
		tokenizedText = []
		for sentence in text:
			# split sentence into tokens(words) and punctuation marks
			# tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", sentence)
			tokens = re.findall(r"[a-zA-Z]+", sentence.lower())
			tokenizedText.append(tokens)    # Append it in the tokenizedText list

		return tokenizedText


	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		# Fill in code here
		tokenizedText = []
        # Function to tokenize each sentence using the Penn Tree Bank Tokenizer
		for sentence in text:
			# Use the Penn Tree Bank Tokenizer to tokenize the sentence
			# It split the sentence and handling punctuation correctly
			# Eg Don't use it. -> ['Don',''', 't','use', 'it', '.']		
			tokens = self.ptb_tokenizer.tokenize(sentence)
			tokenizedText.append(tokens)	

		return tokenizedText


	def spacyTokenizer(self, text):
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		# Fill in code here
		tokenizedText = []

		for sentence in text:
			# Let spacy break the sentence into tokens
            # It handle punctuation and correctly split the sentence into tokens
			doc = self.nlp(sentence)
			tokens = []
            # Save the token in a list
			for token in doc:
				tokens.append(token.text)

			tokenizedText.append(tokens)

		return tokenizedText
