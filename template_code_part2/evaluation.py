from util import *

# Add your import statements here
import math


class Evaluation():

	def _build_relevance_lookup(self, qrels):
		relevance_lookup = {}
		for item in qrels:
			query_id = int(item.get("query number", item.get("query_num")))
			doc_id = int(item["id"])
			position = int(item.get("position", item.get("relevance", 0)))
			# Convert Cranfield position 1-4 into graded relevance 4-1.
			relevance = 5 - position if 1 <= position <= 4 else 0
						
			if query_id not in relevance_lookup:
				relevance_lookup[query_id] = {}
			relevance_lookup[query_id][doc_id] = relevance
		return relevance_lookup

	def _get_relevant_doc_ids(self, qrels, query_id):
		relevant_doc_ids = []
		for item in qrels:
			item_query_id = int(item.get("query number", item.get("query_num")))
			if item_query_id != int(query_id):
				continue
			# In this assignment, positions 1 to 4 are treated as relevant.
			if 1 <= int(item.get("position", item.get("relevance", 0))) <= 4:
				relevant_doc_ids.append(int(item["id"]))
		return relevant_doc_ids

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query
		"""

		if k <= 0:
			return 0.0
		relevant = set(true_doc_IDs)
		retrieved = query_doc_IDs_ordered[:k]
		# Count how many retrieved documents are truly relevant.
		relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
		precision = relevant_retrieved / float(k)

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		precision_values = []
		for ordered_docs, query_id in zip(doc_IDs_ordered, query_ids):
			true_doc_IDs = self._get_relevant_doc_ids(qrels, query_id)
			precision_values.append(self.queryPrecision(ordered_docs, query_id, true_doc_IDs, k))
		meanPrecision = sum(precision_values) / float(len(precision_values) or 1)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query
		"""
		relevant = set(true_doc_IDs)
		if not relevant:
			return 0.0
		retrieved = query_doc_IDs_ordered[:k]
		# Recall divides relevant retrieved documents by all relevant documents.
		relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
		recall = relevant_retrieved / float(len(relevant))

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		recall_values = []
		for ordered_docs, query_id in zip(doc_IDs_ordered, query_ids):
			true_doc_IDs = self._get_relevant_doc_ids(qrels, query_id)
			recall_values.append(self.queryRecall(ordered_docs, query_id, true_doc_IDs, k))
		meanRecall = sum(recall_values) / float(len(recall_values) or 1)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query
		"""
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		# beta = 0.5 gives more weight to precision than recall.
		beta = 0.5
		beta_sq = beta * beta
		if precision == 0.0 and recall == 0.0:
			return 0.0
		fscore = ((1 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		fscore_values = []
		for ordered_docs, query_id in zip(doc_IDs_ordered, query_ids):
			true_doc_IDs = self._get_relevant_doc_ids(qrels, query_id)
			fscore_values.append(self.queryFscore(ordered_docs, query_id, true_doc_IDs, k))
		meanFscore = sum(fscore_values) / float(len(fscore_values) or 1)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query
		"""
		relevance_lookup = true_doc_IDs
		dcg = 0.0
		for rank, doc_id in enumerate(query_doc_IDs_ordered[:k], start=1):
			relevance = relevance_lookup.get(doc_id, 0)
			gain = float(relevance)
			# Standard nDCG discounts rank i by log2(i + 1).
			dcg += gain / math.log(rank + 1, 2)

		ideal_relevances = sorted(relevance_lookup.values(), reverse=True)[:k]
		idcg = 0.0
		for rank, relevance in enumerate(ideal_relevances, start=1):
			gain = float(relevance)
			# Use the same discount for the ideal ranking.
			idcg += gain / math.log(rank + 1, 2)

		if idcg == 0.0:
			return 0.0
		nDCG = dcg / idcg

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		relevance_lookup = self._build_relevance_lookup(qrels)
		ndcg_values = []
		for ordered_docs, query_id in zip(doc_IDs_ordered, query_ids):
			ndcg_values.append(self.queryNDCG(ordered_docs, query_id, relevance_lookup.get(int(query_id), {}), k))
		meanNDCG = sum(ndcg_values) / float(len(ndcg_values) or 1)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)
		"""
		relevant = set(true_doc_IDs)
		if not relevant:
			return 0.0

		precision_sum = 0.0
		hits = 0
		for rank, doc_id in enumerate(query_doc_IDs_ordered[:k], start=1):
			if doc_id in relevant:
				# Add precision at each rank where a relevant document is found.
				hits += 1
				precision_sum += hits / float(rank)
		avgPrecision = precision_sum / float(len(relevant))

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries
		"""
		ap_values = []
		for ordered_docs, query_id in zip(doc_IDs_ordered, query_ids):
			true_doc_IDs = self._get_relevant_doc_ids(q_rels, query_id)
			ap_values.append(self.queryAveragePrecision(ordered_docs, query_id, true_doc_IDs, k))
		meanAveragePrecision = sum(ap_values) / float(len(ap_values) or 1)

		return meanAveragePrecision



	def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of reciprocal rank for a single query
		"""

		relevant = set(true_doc_IDs)
		for rank, doc_id in enumerate(query_doc_IDs_ordered[:k], start=1):
			if doc_id in relevant:
				# Return the inverse rank of the first relevant document.
				return 1.0 / float(rank)

		reciprocalRank = 0.0

		return reciprocalRank


	def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of Mean Reciprocal Rank (MRR)
		averaged over all queries
		"""

		rr_values = []
		for ordered_docs, query_id in zip(doc_IDs_ordered, query_ids):
			true_doc_IDs = self._get_relevant_doc_ids(qrels, query_id)
			rr_values.append(self.queryReciprocalRank(ordered_docs, query_id, true_doc_IDs, k))

		meanReciprocalRank = sum(rr_values) / float(len(rr_values) or 1)

		return meanReciprocalRank
