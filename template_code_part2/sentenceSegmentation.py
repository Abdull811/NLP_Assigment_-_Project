from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download("punkt_tab")

class SentenceSegmentation():

	def __init__(self):
		# Load spaCy model (students may use this if needed)
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
         

		segmentedText = None

		# Fill in code here
		# if input text is empty, return an empty list
		if not text:
			return []
		
		# Split text using regex rule (?<=[.!?]) split after '.', '!', or '?'
		segments = re.split(r'(?<=[.!?])\s+', text.strip())
        
		segmentedText = []
		# Loop through segments 
		for sentence in segments:
			# Remove extra spaces
			sentence = sentence.strip()

			# only add non-empty sentences 
			if sentence:
				segmentedText.append(sentence)
		
		# Return the list of segmented sentences
		return segmentedText


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
        
		segmentedText = None
		# Fill in code here
		# Return empty list if input text is empty
		if not text:
			return []
		
		# Use nltk's sent_tokenize to split text into sentences
		segmentedText = sent_tokenize(text.strip())

		return segmentedText


	def spacySegmenter(self, text):
		"""
		Sentence Segmentation using spaCy

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None
	
		# Fill in code here
		if not text:
			return []
		
		# Process the text using spacy 
		doc = self.nlp(text)

		segmentedText = []
		# save the sentence in a list
		for sent in doc.sents:
			segmentedText.append(sent.text.strip())

		return segmentedText
