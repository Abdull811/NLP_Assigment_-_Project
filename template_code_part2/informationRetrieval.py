from util import *

# Add your import statements here
import math
from collections import Counter


class InformationRetrieval():

	def __init__(self):

		self.index = None
		self.doc_ids = []
		self.doc_vectors = {}
		self.doc_norms = {}
		self.idf = {}
		self.vocabulary = set()
		self.doc_term_counts = {}
		# Pseudo-relevance feedback settings. The system assumes the first
		# few retrieved documents are useful, then expands the query vector.
		self.feedback_docs = 2
		self.query_alpha = 1.4
		self.feedback_beta = 0.4

	def _flatten(self, text):
		terms = []

		for sentence in text:
			terms.extend(sentence)

		return terms
    
	def _get_terms(self, text):
		unigrams = self._flatten(text)
		bigrams = []

		for sentence in text:
			for i in range(len(sentence) - 1):
				bigrams.append(sentence[i] + "_" + sentence[i + 1])

		return unigrams + bigrams

		
	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		self.doc_ids = list(docIDs)
		self.doc_term_counts = {}
		document_frequency = Counter()

		for doc_id, doc in zip(docIDs, docs):
			terms = self._flatten(doc)
			# terms = self._get_terms(doc)
			term_counts = Counter(terms)
			self.doc_term_counts[doc_id] = term_counts

			for term in term_counts.keys():
				document_frequency[term] += 1

		num_docs = float(len(docIDs))
		self.vocabulary = set(document_frequency.keys())

		# Smoothed IDF
		self.idf = {
			term: math.log((num_docs + 1.0) / (document_frequency[term] + 1.0)) + 1.0
			for term in document_frequency
		}

		self.doc_vectors = {}
		self.doc_norms = {}

		for doc_id, term_counts in self.doc_term_counts.items():
			vector = {}

			for term, count in term_counts.items():
				tf = 1.0 + math.log(count)
				vector[term] = tf * self.idf[term]

			norm = math.sqrt(sum(weight * weight for weight in vector.values()))
			self.doc_vectors[doc_id] = vector
			self.doc_norms[doc_id] = norm if norm != 0 else 1e-6

		self.index = {
			"doc_term_counts": self.doc_term_counts,
			"document_frequency": dict(document_frequency),
			"idf": self.idf,
			"doc_vectors": self.doc_vectors,
			"doc_norms": self.doc_norms
		}


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		def score_documents(query_vector):
			query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
			if query_norm == 0:
				query_norm = 1e-6

			scores = []

			for doc_id in self.doc_ids:
				doc_vector = self.doc_vectors[doc_id]
				dot_product = 0.0

				for term, q_weight in query_vector.items():
					if term in doc_vector:
						dot_product += q_weight * doc_vector[term]

				score = dot_product / (query_norm * self.doc_norms[doc_id])
				scores.append((doc_id, score))

			scores.sort(key=lambda x: (-x[1], x[0]))
			return scores

		doc_IDs_ordered = []

		for query in queries:
			query_terms = self._flatten(query)
			# query_terms = self._get_terms(query)
			query_term_counts = Counter(term for term in query_terms if term in self.vocabulary)

			if not query_term_counts:
				doc_IDs_ordered.append(list(self.doc_ids))
				continue

			query_vector = {}
			for term, count in query_term_counts.items():
				tf = 1.0 + math.log(count)
				query_vector[term] = tf * self.idf[term]

			scores = score_documents(query_vector)

			# Rocchio-style pseudo-relevance feedback:
			# expand the original query using the average TF-IDF vector of the
			# top-ranked documents, then rank again.
			# Keep the original query slightly stronger than the feedback terms.
			expanded_query_vector = {
				term: self.query_alpha * weight
				for term, weight in query_vector.items()
			}
			feedback_doc_ids = [doc_id for doc_id, _ in scores[:self.feedback_docs]]

			for doc_id in feedback_doc_ids:
				for term, weight in self.doc_vectors[doc_id].items():
					expanded_query_vector[term] = (
						expanded_query_vector.get(term, 0.0)
						+ (self.feedback_beta * weight / float(self.feedback_docs))
					)

			scores = score_documents(expanded_query_vector)
			doc_IDs_ordered.append([doc_id for doc_id, _ in scores])

		return doc_IDs_ordered
