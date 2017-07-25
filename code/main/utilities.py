import numpy as np
import csv
import configuration as config
import random
import heapq
import re
import numpy as np
import scipy.stats


################################################################ Beam search data structures
class TopN(object):
	"""Maintains the top n elements of an incrementally provided set."""

	def __init__(self, n):
		self._n = n
		self._data = []

	def size(self):
		assert self._data is not None
		return len(self._data)

	def push(self, x):
		"""Pushes a new element."""
		assert self._data is not None
		if len(self._data) < self._n:
			heapq.heappush(self._data, x)
		else:
			heapq.heappushpop(self._data, x)

	def extract(self, sort=False):
		"""Extracts all elements from the TopN. This is a destructive operation.
		The only method that can be called immediately after extract() is reset().
		Args:
			sort: Whether to return the elements in descending sorted order.
		Returns:
			A list of data; the top n elements provided to the set.
		"""
		assert self._data is not None
		data = self._data
		self._data = None
		if sort:
			data.sort(reverse=True)
		return data

	def reset(self):
		"""Returns the TopN to an empty state."""
		self._data = []

################################################################

class OutputSentence(object):
	"""Represents a complete or partial caption."""

	def __init__(self, sentence, state, logprob, score, metadata=None):
		"""Initializes the Caption.
		Args:
			sentence: List of word ids in the caption.
			state: Model state after generating the previous word.
			logprob: Log-probability of the caption.
			score: Score of the caption.
			metadata: Optional metadata associated with the partial sentence. If not
				None, a list of strings with the same length as 'sentence'.
		"""
		self.sentence = sentence
		self.state = state
		self.logprob = logprob
		self.score = score
		self.metadata = metadata

	def __cmp__(self, other):
		"""Compares Captions by score."""
		assert isinstance(other, OutputSentence)
		if self.score == other.score:
			return 0
		elif self.score < other.score:
			return -1
		else:
			return 1
	
	# For Python 3 compatibility (__cmp__ is deprecated).
	def __lt__(self, other):
		assert isinstance(other, OutputSentence)
		return self.score < other.score
	
	# Also for Python 3 compatibility.
	def __eq__(self, other):
		assert isinstance(other, OutputSentence)
		return self.score == other.score

################################################################

def sampleFromDistribution(vals):
		p = random.random()
		s=0.0
		for i,v in enumerate(vals):
				s+=v
				if s>=p:
						return i
		return len(vals)-1


################################################################
import os
def getBlue(validOutFile_name, original_data_path, BLEUOutputFile_path, decoder_outputs_inference, decoder_ground_truth_outputs, preprocessing_obj, verbose=False):
	validOutFile=open(validOutFile_name,"w")
	for outputLine,groundLine in zip(decoder_outputs_inference, decoder_ground_truth_outputs):
		if verbose:
			print outputLine
		outputLine=preprocessing_obj.fromIdxSeqToVocabSeq(outputLine)
		if "sentend" in outputLine:
			outputLine=outputLine[:outputLine.index("sentend")]
		if verbose:
			print outputLine
			print preprocessing_obj.fromIdxSeqToVocabSeq(groundLine)
		outputLine=" ".join(outputLine)+"\n"
		validOutFile.write(outputLine)
	validOutFile.close()

	BLEUOutput=os.popen("perl multi-bleu.perl -lc " + original_data_path + " < " + validOutFile_name).read()
	BLEUOutputFile=open(BLEUOutputFile_path,"w")
	BLEUOutputFile.write(BLEUOutput)
	BLEUOutputFile.close()



#############################################################
def preprocessText(list_of_sentences, preprocessing_obj):

	preprocessing = preprocessing_obj
	all_txt_tokenized = [ txt.split() for txt in list_of_sentences] #TO DO: nltk tokenization
	all_txt_indexed = [ ]
	for txt in all_txt_tokenized:
		tmp = []
		tmp.append(preprocessing.word_to_idx[preprocessing.sent_start])
		for w in txt:
			if w not in preprocessing.word_to_idx:
				w = preprocessing.unknown_word
			tmp.append( preprocessing.word_to_idx[w] )
		tmp.append(preprocessing.word_to_idx[preprocessing.sent_end])
		all_txt_indexed.append(tmp)

	return all_txt_indexed