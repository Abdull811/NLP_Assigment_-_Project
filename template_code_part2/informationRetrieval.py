from util import *

# Add your import statements here
import math
from collections import Counter, defaultdict


class InformationRetrieval():

	def __init__(self):

		self.index = None
		self.doc_ids = []
		self.doc_vectors = {}
		self.doc_norms = {}
		self.term_doc_weights = {}
		self.idf = {}
		self.vocabulary = set()
		self.doc_term_counts = {}
		# Pseudo-relevance feedback settings. The system assumes the first
		# few retrieved documents are useful, then expands the query vector.
		self.feedback_docs = 1
		self.non_feedback_docs = 1
		self.query_alpha = 1.0
		self.feedback_beta = 0.75
		self.feedback_gamma = 0.15
		self.use_feedback =  True # False


	def _flatten(self, text):
		terms = []

		# Convert list of sentences into one list of terms.
		for sentence in text:
			terms.extend(sentence)

		return terms
    
	def _get_terms(self, text):
		unigrams = self._flatten(text)
		bigrams = []

		# Build bigrams by joining each pair of neighboring terms.
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

		# Count terms in every document and collect document frequencies.
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
		self.term_doc_weights = defaultdict(dict)

		# Create TF-IDF vectors and store each term's posting weights.
		for doc_id, term_counts in self.doc_term_counts.items():
			vector = {}

			for term, count in term_counts.items():
				tf = 1.0 + math.log(count)
				weight = tf * self.idf[term]
				vector[term] = weight
				self.term_doc_weights[term][doc_id] = weight

			# Store document norm for cosine similarity.
			norm = math.sqrt(sum(weight * weight for weight in vector.values()))
			self.doc_vectors[doc_id] = vector
			self.doc_norms[doc_id] = norm if norm != 0 else 1e-6

		self.index = {
			"doc_term_counts": self.doc_term_counts,
			"document_frequency": dict(document_frequency),
			"idf": self.idf,
			"doc_vectors": self.doc_vectors,
			"doc_norms": self.doc_norms,
			"term_doc_weights": dict(self.term_doc_weights)
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
			# Compute query length for cosine similarity.
			query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
			if query_norm == 0:
				query_norm = 1e-6

			dot_products = defaultdict(float)

			# Use the inverted TF-IDF index so only documents sharing query
			# terms are scored. Documents with zero score are appended later.
			for term, q_weight in query_vector.items():
				for doc_id, doc_weight in self.term_doc_weights.get(term, {}).items():
					dot_products[doc_id] += q_weight * doc_weight

			scores = []
			scored_doc_ids = set(dot_products.keys())
			for doc_id, dot_product in dot_products.items():
				# Final cosine score = dot product divided by both vector norms.
				score = dot_product / (query_norm * self.doc_norms[doc_id])
				scores.append((doc_id, score))

			# Sort by higher score first; use document id to break ties.
			scores.sort(key=lambda x: (-x[1], x[0]))
			zero_score_docs = [(doc_id, 0.0) for doc_id in self.doc_ids if doc_id not in scored_doc_ids]
			return scores + zero_score_docs

		doc_IDs_ordered = []

		for query in queries:
			query_terms = self._flatten(query)
			# query_terms = self._get_terms(query)
			# Ignore query terms that never appear in the document collection.
			query_term_counts = Counter(term for term in query_terms if term in self.vocabulary)

			if not query_term_counts:
				doc_IDs_ordered.append(list(self.doc_ids))
				continue

			# Build the TF-IDF vector for the query.
			query_vector = {}
			for term, count in query_term_counts.items():
				tf = 1.0 + math.log(count)
				query_vector[term] = tf * self.idf[term]

			scores = score_documents(query_vector)

			if not self.use_feedback:
				doc_IDs_ordered.append([doc_id for doc_id, _ in scores])
				continue

			# Rocchio-style pseudo-relevance feedback:
			# expand the original query using the average TF-IDF vector of the
			# top-ranked documents and reduce terms from bottom-ranked documents,
			# then rank again.
			# Keep the original query slightly stronger than the feedback terms.
			expanded_query_vector = {
				term: self.query_alpha * weight
				for term, weight in query_vector.items()
			}
			feedback_doc_ids = [doc_id for doc_id, _ in scores[:self.feedback_docs]]
			non_feedback_doc_ids = [doc_id for doc_id, _ in scores[-self.non_feedback_docs:]]

			for doc_id in feedback_doc_ids:
				for term, weight in self.doc_vectors[doc_id].items():
					expanded_query_vector[term] = (
						expanded_query_vector.get(term, 0.0)
						+ (self.feedback_beta * weight / float(self.feedback_docs))
					)

			for doc_id in non_feedback_doc_ids:
				for term, weight in self.doc_vectors[doc_id].items():
					expanded_query_vector[term] = (
						expanded_query_vector.get(term, 0.0)
						- (self.feedback_gamma * weight / float(self.non_feedback_docs))
					)

			# Remove negative weights because cosine scoring expects positive term weights.
			expanded_query_vector = {
				term: weight
				for term, weight in expanded_query_vector.items()
				if weight > 0.0
			}

			scores = score_documents(expanded_query_vector)
			doc_IDs_ordered.append([doc_id for doc_id, _ in scores])

		return doc_IDs_ordered


	def setFeedback(self, enabled):
		"""
		Enable or disable pseudo-relevance feedback.
		This lets main.py compare the basic TF-IDF VSM against the improved
		system without changing code between runs.
		"""

		self.use_feedback = enabled
