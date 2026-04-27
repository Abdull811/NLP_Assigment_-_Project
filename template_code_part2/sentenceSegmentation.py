from util import *

# Add your import statements here
import re
import nltk
from nltk.tokenize import sent_tokenize

for resource_path, package_name in [
	("tokenizers/punkt", "punkt"),
	("tokenizers/punkt_tab", "punkt_tab"),
]:
	try:
		nltk.data.find(resource_path)
	except LookupError:
		nltk.download(package_name)

class SentenceSegmentation():

	def __init__(self):
		# Load spaCy only if the spaCy segmenter is selected.
		self.nlp = None

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
		if self.nlp is None:
			import spacy
			self.nlp = spacy.load("en_core_web_sm")
		doc = self.nlp(text)

		segmentedText = []
		# save the sentence in a list
		for sent in doc.sents:
			segmentedText.append(sent.text.strip())

		return segmentedText
