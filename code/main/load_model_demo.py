import tensorflow as tf
# Set seed for reproducability
tf.set_random_seed(1)
import numpy as np
np.random.seed(1)

from keras.preprocessing.sequence import pad_sequences
import configuration as config
import pickle
import sys
import mt_model as models
import utilities as datasets
import utilities
import mt_solver as solver
from prepro import PreProcessing
from tensorflow.contrib import rnn

########################

data_src = config.data_dir

def main(saved_model_path, inference_type="greedy"):
	
	# params
	params = {}
	params['embeddings_dim'] =  config.embeddings_dim
	params['lstm_cell_size'] = config.lstm_cell_size
	params['max_input_seq_length'] = config.max_input_seq_length
	params['max_output_seq_length'] = config.max_output_seq_length-1 #inputs are all but last element, outputs are al but first element
	params['batch_size'] = config.batch_size
	params['pretrained_embeddings'] = config.use_pretrained_embeddings
	params['pretrained_embeddings'] = True
	params['share_encoder_decoder_embeddings'] = config.share_encoder_decoder_embeddings
	params['use_pointer'] = config.use_pointer
	params['pretrained_embeddings_path'] = config.pretrained_embeddings_path
	params['pretrained_embeddings_are_trainable'] = config.pretrained_embeddings_are_trainable
	params['use_additional_info_from_pretrained_embeddings'] = config.use_additional_info_from_pretrained_embeddings
	params['max_vocab_size'] = config.max_vocab_size
	params['do_vocab_pruning'] = config.do_vocab_pruning
	params['use_reverse_encoder'] = config.use_reverse_encoder
	params['use_sentinel_loss'] =config.use_sentinel_loss
	params['lambd'] = config.lambd
	params['use_context_for_out'] = config.use_context_for_out

	params['batch_size'] = 32

	print "PARAMS:"
	for key,value in params.items():
		print " -- ",key," = ",value
	buckets = {  0:{'max_input_seq_length':params['max_input_seq_length'], 'max_output_seq_length':params['max_output_seq_length']} }
	#print "buckets = ",buckets
	
	#	data = pickle.load(open(data_src + "data.obj","r") )
	preprocessing = pickle.load(open(data_src + "preprocessing.obj","r") )

	params['vocab_size'] = preprocessing.vocab_size
	params['preprocessing'] = preprocessing

	saved_model_path = saved_model_path
	print "saved_model_path = ",saved_model_path
	print "inference_type = ",inference_type
	params['saved_model_path'] = saved_model_path

	if params['pretrained_embeddings']:
		pretrained_embeddings = pickle.load(open(params['pretrained_embeddings_path'],"r"))
		word_to_idx = preprocessing.word_to_idx
		encoder_embedding_matrix = np.random.rand( params['vocab_size'], params['embeddings_dim'] )
		decoder_embedding_matrix = np.random.rand( params['vocab_size'], params['embeddings_dim'] )
		not_found_count = 0
		for token,idx in word_to_idx.items():
			if token in pretrained_embeddings:
				encoder_embedding_matrix[idx]=pretrained_embeddings[token]
				decoder_embedding_matrix[idx]=pretrained_embeddings[token]
			else:
				if not_found_count<10:
					print "No pretrained embedding for (only first 10 such cases will be printed. other prints are suppressed) ",token
				not_found_count+=1
		#print "not found count = ", not_found_count 
		params['encoder_embeddings_matrix'] = encoder_embedding_matrix 
		params['decoder_embeddings_matrix'] = decoder_embedding_matrix 

		if params['use_additional_info_from_pretrained_embeddings']:
			additional_count=0
			tmp=[]
			for token in pretrained_embeddings:
				if token not in preprocessing.word_to_idx:
					preprocessing.word_to_idx[token] = preprocessing.word_to_idx_ctr
					preprocessing.idx_to_word[preprocessing.word_to_idx_ctr] = token
					preprocessing.word_to_idx_ctr+=1
					tmp.append(pretrained_embeddings[token])
					additional_count+=1
			#print "additional_count = ",additional_count
			params['vocab_size'] = preprocessing.word_to_idx_ctr
			tmp = np.array(tmp)
			encoder_embedding_matrix = np.vstack([encoder_embedding_matrix,tmp])
			decoder_embedding_matrix = np.vstack([decoder_embedding_matrix,tmp])
			#print "decoder_embedding_matrix.shape ",decoder_embedding_matrix.shape
			#print "New vocab size = ",params['vocab_size']

	rnn_model = solver.Solver(params, buckets=None, mode='inference')
	_ = rnn_model.getModel(params, mode='inference', reuse=False, buckets=None)
	
	### preprocessing.word_to_idx - use to convert arbit sentence to seq of ids. first do some preprocessing though - everythino to lower. use nltk tokenizer. unknown word converte to UNK
	### outputLine=preprocessing_obj.fromIdxSeqToVocabSeq(outputLine) - to convert output idx sequences to sentence - can also use post processing - highest attenttion input word can be used in place of UNK outputs

	all_txt = ["Is this your book ?"]
	all_txt_tokenized = [ txt.split() for txt in all_txt] #TO DO: tokenization
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

	sequences_input = pad_sequences(all_txt_indexed, maxlen=config.max_input_seq_length, padding='pre', truncating='post')

	decoder_outputs_inference, _ = rnn_model.solveAll(params, sequences_input, None, preprocessing.idx_to_word, inference_type=inference_type)
	print decoder_outputs_inference      
	return

	#val
	data = pickle.load(open(data_src + "data.obj","r") )
	val = data['valid']
	val_encoder_inputs, val_decoder_inputs, val_decoder_outputs, val_decoder_outputs_matching_inputs = val
	#print "val_encoder_inputs = ",val_encoder_inputs
	if len(val_decoder_outputs.shape)==3:
		val_decoder_outputs=np.reshape(val_decoder_outputs, (val_decoder_outputs.shape[0], val_decoder_outputs.shape[1]))
	decoder_outputs_inference, decoder_ground_truth_outputs = rnn_model.solveAll(params, val_encoder_inputs, val_decoder_outputs, preprocessing.idx_to_word, inference_type=inference_type)        			   
	validOutFile_name = saved_model_path+".valid.output"
	original_data_path = data_src + "valid.original.nltktok"
	BLEUOutputFile_path = saved_model_path + ".valid.BLEU"
	utilities.getBlue(validOutFile_name, original_data_path, BLEUOutputFile_path, decoder_outputs_inference, decoder_ground_truth_outputs, preprocessing)

	'''

	#test
	test_encoder_inputs, test_decoder_inputs, test_decoder_outputs, test_decoder_outputs_matching_inputs = test
	if len(test_decoder_outputs.shape)==3:
		test_decoder_outputs=np.reshape(test_decoder_outputs, (test_decoder_outputs.shape[0], test_decoder_outputs.shape[1]))
	decoder_outputs_inference, decoder_ground_truth_outputs = rnn_model.solveAll(params, test_encoder_inputs, test_decoder_outputs, preprocessing.idx_to_word, inference_type=inference_type)
	validOutFile_name = saved_model_path+".test.output"
	original_data_path = data_src + "test.original.nltktok"
	BLEUOutputFile_path = saved_model_path + ".test.BLEU"
	utilities.getBlue(validOutFile_name, original_data_path, BLEUOutputFile_path, decoder_outputs_inference, decoder_ground_truth_outputs, preprocessing)

	'''
if __name__ == "__main__":
	main("./tmp/test5.ckpt")

