from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


class StopwordRemoval():

	def __init__(self):
		# Prepare the list of stopwords
		self.stop_words = set(stopwords.words("english"))

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = None

		#Fill in code here
		stopwordRemovedText = []
        # Function that iterate through each sentence and remove stopwords from it
		for sentence in text:
			filtered_sentence = []
			for token in sentence:
				# Check if the token is not stopword
				# If it's not a stopword, add it to the filtered sentence
				# if token.lower() not in self.stop_words:
				if token.lower() not in self.stop_words and len(token) > 2:
					filtered_sentence.append(token)
			# Save the cleaned sentence
			stopwordRemovedText.append(filtered_sentence)		

		return stopwordRemovedText
	