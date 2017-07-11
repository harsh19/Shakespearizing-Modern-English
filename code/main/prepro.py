from keras.preprocessing.sequence import pad_sequences
import numpy as np
import configuration as config
import pickle

class PreProcessing:

	def __init__(self):
		self.unknown_word = "UNK".lower()
		self.sent_start = "SENTSTART".lower()
		self.sent_end = "SENTEND".lower()
		self.pad_word = "PADWORD".lower()
		self.special_tokens = [self.sent_start, self.sent_end, self.pad_word, self.unknown_word]

		self.word_counters, self.word_to_idx, self.word_to_idx_ctr, self.idx_to_word = self.initVocabItems()

	def initVocabItems(self):

		word_counters = {}
		word_to_idx = {}
		word_to_idx_ctr = 0 
		idx_to_word = {}

		word_to_idx[self.pad_word] = word_to_idx_ctr # 0 is for padword
		idx_to_word[word_to_idx_ctr]=self.pad_word
		word_counters[self.pad_word] = 1
		word_to_idx_ctr+=1
		
		word_to_idx[self.sent_start] = word_to_idx_ctr
		word_counters[self.sent_start] = 1
		idx_to_word[word_to_idx_ctr]=self.sent_start
		word_to_idx_ctr+=1

		word_to_idx[self.sent_end] = word_to_idx_ctr
		word_counters[self.sent_end] = 1
		idx_to_word[word_to_idx_ctr]=self.sent_end		
		word_to_idx_ctr+=1
		
		word_counters[self.unknown_word] = 1
		word_to_idx[self.unknown_word] = word_to_idx_ctr
		idx_to_word[word_to_idx_ctr]=self.unknown_word		
		word_to_idx_ctr+=1

		return word_counters, word_to_idx, word_to_idx_ctr, idx_to_word

	def pad_sequences_my(sequences, maxlen, padding='post', truncating='post'):
		ret=[]
		for sequence in sequences:
			if len(sequence)>=maxlen:
				sequence=sequence[:maxlen]
			else:
				if padding=='post':
					sequence = sequence + [0]*(maxlen - len(sequence))
				else:
					sequence = [0]*(maxlen - len(sequence)) + sequence
			ret.append(sequence)
		return np.array(ret)

	def preprocess(self, text_rows):
		return [row.strip().lower().split(' ') for row in text_rows]

	def loadVocab(self, split):

		print "======================================================= loadData: split = ",split
		inp_src = config.data_dir + split + ".original" + ".nltktok" #".modern"
		out_src = config.data_dir + split + ".modern" + ".nltktok" #".original"
		inp_data = open(inp_src,"r").readlines()
		out_data = open(out_src,"r").readlines()
		
		inputs = self.preprocess(inp_data)
		outputs = self.preprocess(out_data)
		
		word_to_idx = self.word_to_idx
		idx_to_word = self.idx_to_word
		word_to_idx_ctr = self.word_to_idx_ctr
		word_counters = self.word_counters

		texts = inputs
		for text in texts:
			for token in text:
				if token not in word_to_idx:
					word_to_idx[token] = word_to_idx_ctr
					idx_to_word[word_to_idx_ctr]=token
					word_to_idx_ctr+=1
					word_counters[token]=0
				word_counters[token]+=1
		texts = outputs
		for text in texts:
			for token in text:
				if token not in word_to_idx:
					word_to_idx[token] = word_to_idx_ctr
					idx_to_word[word_to_idx_ctr]=token
					word_to_idx_ctr+=1
					word_counters[token]=0
				word_counters[token]+=1

		self.word_to_idx = word_to_idx
		self.idx_to_word = idx_to_word
		self.vocab_size = len(word_to_idx)
		self.word_to_idx_ctr = word_to_idx_ctr
		self.word_counters = word_counters

	def pruneVocab(self, max_vocab_size):
		word_to_idx = self.word_to_idx
		idx_to_word = self.idx_to_word
		word_to_idx_ctr = self.word_to_idx_ctr
		word_counters = self.word_counters

		tmp_word_counters, tmp_word_to_idx, tmp_word_to_idx_ctr, tmp_idx_to_word = self.initVocabItems()
		print "** ",tmp_idx_to_word[1]

		print "vocab size before pruning = ", len(word_to_idx)
		top_items = sorted( word_counters.items(), key=lambda x:-x[1] )[:max_vocab_size]
		for token_count in top_items:
			token=token_count[0]
			if token in self.special_tokens:
				continue
			tmp_word_to_idx[token] = tmp_word_to_idx_ctr
			tmp_idx_to_word[tmp_word_to_idx_ctr] = token
			tmp_word_to_idx_ctr+=1
		print "** ",tmp_idx_to_word[9947]

		self.word_to_idx = tmp_word_to_idx
		self.idx_to_word = tmp_idx_to_word
		self.vocab_size = len(tmp_word_to_idx)
		self.word_to_idx_ctr = tmp_word_to_idx_ctr
		print "vocab size after pruning = ", self.vocab_size


	def loadData(self, split):

		print "======================================================= loadData: split = ",split
		inp_src = config.data_dir + split + ".original" + ".nltktok" #".modern"
		out_src = config.data_dir + split + ".modern" + ".nltktok" #".original"
		inp_data = open(inp_src,"r").readlines()
		out_data = open(out_src,"r").readlines()
		
		inputs = self.preprocess(inp_data)
		outputs = self.preprocess(out_data)
		
		word_to_idx = self.word_to_idx
		idx_to_word = self.idx_to_word
		word_to_idx_ctr = self.word_to_idx_ctr

		# generate sequences
		sequences_input = [] 		
		sequences_output = [] 

		texts = inputs
		for text in texts:
			tmp = [word_to_idx[self.sent_start]]
			for token in text:
				if token not in word_to_idx:
					tmp.append(word_to_idx[self.unknown_word])
				else:
					tmp.append(word_to_idx[token])
			tmp.append(word_to_idx[self.sent_end])
			sequences_input.append(tmp)

		texts = outputs
		for text in texts:
			tmp = [word_to_idx[self.sent_start]]
			for token in text:
				if token not in word_to_idx:
					tmp.append(word_to_idx[self.unknown_word])
				else:
					tmp.append(word_to_idx[token])
			tmp.append(word_to_idx[self.sent_end])
			sequences_output.append(tmp)

		# pad sequences
		# sequences_input, sequences_output = padAsPerBuckets(sequences_input, sequences_output)
		sequences_input = pad_sequences(sequences_input, maxlen=config.max_input_seq_length, padding='pre', truncating='post')
		sequences_output = pad_sequences(sequences_output, maxlen=config.max_output_seq_length, padding='post', truncating='post')

		print "Printing few sample sequences... "
		print sequences_input[0],":", self.fromIdxSeqToVocabSeq(sequences_input[0]), "---", sequences_output[0], ":", self.fromIdxSeqToVocabSeq(sequences_output[0])
		print sequences_input[113], sequences_output[113]
		print "================================="

		return sequences_input, sequences_output

	def fromIdxSeqToVocabSeq(self, seq):
		return [self.idx_to_word[x] for x in seq]

	def prepareMTData(self, sequences, seed=123, do_shuffle=False):
		inputs, outputs = sequences

		decoder_inputs = np.array( [ sequence[:-1] for sequence in outputs ] )
		#decoder_outputs = np.array( [ np.expand_dims(sequence[1:],-1) for sequence in outputs ] )
		decoder_outputs = np.array( [ sequence[1:] for sequence in outputs ] )
		matching_input_token = []
		for cur_outputs, cur_inputs in zip(decoder_outputs, inputs):
			tmp = []
			for output_token in cur_outputs:
				idx = np.zeros(len(cur_inputs), dtype=np.float32)
				for j,token in enumerate(cur_inputs):
					if token <= 3:  #==self.word_to_idx[self.pad_word]:
						continue
					if token == output_token:
						idx[j]=1.0
				tmp.append(idx)
			matching_input_token.append(tmp)
		matching_input_token = np.array(matching_input_token)
		encoder_inputs = np.array(inputs)

		'''
		print "-------------------------------------------"
		print matching_input_token[0]
		print decoder_outputs[0]
		print encoder_inputs[0]
		print "-------------------------------------------"
		print matching_input_token[1]
		print decoder_outputs[1]
		print encoder_inputs[1]
		print "-------------------------------------------"		
		'''

		if do_shuffle:
			#shuffling
			indices = np.arange(encoder_inputs.shape[0])
			np.random.seed(seed)
			np.random.shuffle(indices)
		print "np.sum(np.sum(np.sum(matching_input_token))) = ",np.sum(np.sum(np.sum(matching_input_token)))
		return encoder_inputs, decoder_inputs, decoder_outputs, matching_input_token

