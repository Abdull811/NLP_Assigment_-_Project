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

	def _flatten(self, text):
		terms = []
		for sentence in text:
			terms.extend(sentence)

		return terms

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
		doc_term_counts = {}
		document_frequency = Counter()

		for doc_id, doc in zip(docIDs, docs):
			terms = self._flatten(doc)
			term_counts = Counter(terms)
			doc_term_counts[doc_id] = term_counts
			document_frequency.update(term_counts.keys())

		num_docs = float(len(docIDs))
		self.vocabulary = set(document_frequency.keys())
		self.idf = {
			term: math.log((num_docs + 1.0) / (document_frequency[term] + 1.0)) + 1.0
			for term in document_frequency
		}

		self.doc_vectors = {}
		self.doc_norms = {}
		for doc_id, term_counts in doc_term_counts.items():
			vector = {}
			total_terms = float(sum(term_counts.values())) or 1.0
			for term, count in term_counts.items():
				# tf = count / total_terms 
				tf = 1 + math.log(count)
				vector[term] = tf * self.idf[term]
			self.doc_vectors[doc_id] = vector
			self.doc_norms[doc_id] = math.sqrt(sum(weight * weight for weight in vector.values())) + 1e-6

		index = {
			"doc_term_counts": doc_term_counts,
			"document_frequency": dict(document_frequency),
			"idf": self.idf,
			"doc_vectors": self.doc_vectors
		}

		self.index = index


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

		doc_IDs_ordered = []

		for query in queries:
			query_terms = self._flatten(query)
			query_term_counts = Counter(term for term in query_terms if term in self.vocabulary)

			if not query_term_counts:
				doc_IDs_ordered.append(list(self.doc_ids))
				continue

			total_terms = float(sum(query_term_counts.values())) or 1.0
			query_vector = {}
			for term, count in query_term_counts.items():
				tf = 1 + math.log(count)
				query_vector[term] = tf * self.idf[term]

			query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) + 1e-6
			scores = []

			for doc_id in self.doc_ids:
				doc_vector = self.doc_vectors[doc_id]
				dot_product = 0.0
				for term, weight in query_vector.items():
					dot_product += weight * doc_vector.get(term, 0.0)

				doc_norm = self.doc_norms[doc_id]
				if query_norm == 0.0 or doc_norm == 0.0:
					score = 0.0
				else:
					score = dot_product / (query_norm * (doc_norm + 1e-6))
				scores.append((doc_id, score))

			scores.sort(key=lambda item: (-item[1], item[0]))
			doc_IDs_ordered.append([doc_id for doc_id, _ in scores])
	
		return doc_IDs_ordered
