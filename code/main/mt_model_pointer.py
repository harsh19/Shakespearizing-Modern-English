import numpy as np
import tensorflow as tf
from utilities import OutputSentence, TopN
import utilities
import configuration as config
from tensorflow.contrib import rnn	

class RNNModel:


	def __init__(self, buckets_dict, mode='training',params={}):
		#print "========== INIT ============= "
		self.use_pointer = params['use_pointer']
		self.use_reverse_encoder = params['use_reverse_encoder']
		if mode=='training':
			self.token_lookup_sequences_decoder_placeholder_list = []
			self.masker_list = []
			self.token_output_sequences_decoder_placeholder_list = []
			self.token_output_sequences_decoder_inpmatch_placeholder_list = []
			self.token_lookup_sequences_placeholder_list = []

			for bucket_num, bucket in buckets_dict.items():
				max_sentence_length = bucket['max_input_seq_length']
				self.token_lookup_sequences_placeholder_list.append( tf.placeholder("int32", [None, max_sentence_length], name="token_lookup_sequences"+str(bucket_num))  )# token_lookup_sequences
				max_sentence_length = bucket['max_output_seq_length']
				self.masker_list.append( tf.placeholder("float32", [None, max_sentence_length], name="masker"+str(bucket_num)) )
				self.token_output_sequences_decoder_placeholder_list.append( tf.placeholder("int32", [None, max_sentence_length], name="token_output_sequences_decoder_placeholder"+str(bucket_num)) )
				self.token_lookup_sequences_decoder_placeholder_list.append( tf.placeholder("int32", [None, max_sentence_length], name="token_lookup_sequences_decoder_placeholder"+str(bucket_num)) ) # token_lookup_sequences

				self.token_output_sequences_decoder_inpmatch_placeholder_list.append( tf.placeholder("float32", [None, bucket['max_output_seq_length'], bucket['max_input_seq_length']], name="token_output_sequences_decoder_inpmatch_placeholder"+str(bucket_num)) )

		#print "========== INIT OVER ============= "


	def _getEncoderInitialState(self,cell, batch_size):
		return cell.zero_state(batch_size, tf.float32)

	def encoderRNN(self, x,lstm_cell_size, batch_size, batch_time_steps, reuse=False, mode='training'):

		num_steps = batch_time_steps #should be equal to x.shape[1]
		inputs = x

		#define lstm cell
		lstm_cell_size = lstm_cell_size
		lstm_cell = rnn.BasicLSTMCell(lstm_cell_size, forget_bias=1.0, state_is_tuple=True, reuse=reuse)
		cell = lstm_cell
		initial_state = self._getEncoderInitialState(cell, batch_size)

		#unrolled lstm 	
		outputs = [] # h values at each time step
		state = initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)
		outputs = tf.stack(outputs) 

		if self.use_reverse_encoder:
			with tf.variable_scope("rev_encoder"):
				rev_lstm_cell = rnn.BasicLSTMCell(lstm_cell_size, forget_bias=1.0, state_is_tuple=True, reuse=reuse)
				cell = rev_lstm_cell
				initial_state = self._getEncoderInitialState(cell, batch_size)
				rev_outputs = [] # h values at each time step
				state = initial_state
				for time_step in range(num_steps-1,-1,-1):
					if time_step < (num_steps-1): tf.get_variable_scope().reuse_variables()
					(cell_output, state) = cell(inputs[:, time_step, :], state)
					rev_outputs = [cell_output] + rev_outputs # reverse encoder
			rev_outputs = tf.stack(rev_outputs) 
			##outputs = tf.concat([outputs, rev_outputs], axis=2)
			outputs = outputs + rev_outputs ## addition. would maintain outout size equal to lstm cell size.

		return outputs

	def getEncoderModel(self, config, bucket_num=-1, mode='training', reuse=False):

		token_vocab_size = config['vocab_size']
		max_sentence_length = config['max_input_seq_length'] # max sequene length of encoder
		embeddings_dim = config['embeddings_dim']
		lstm_cell_size = config['lstm_cell_size']

		#placeholders
		if mode=='training':
			token_lookup_sequences_placeholder = token_lookup_sequences_placeholder = self.token_lookup_sequences_placeholder_list[bucket_num]
		elif mode=='inference':
			self.token_lookup_sequences_placeholder_inference = token_lookup_sequences_placeholder = tf.placeholder("int32", [None, max_sentence_length], name="token_lookup_sequences") # token_lookup_sequences
		
		#get embeddings
		if reuse:
			token_emb_mat = self.encoder_token_emb_mat
		else:
			pretrained_embeddings=None
			if config['pretrained_embeddings']:
				pretrained_embeddings = config['encoder_embeddings_matrix']
			self.encoder_token_emb_mat = token_emb_mat = self.initEmbeddings('emb_encoder', token_vocab_size, embeddings_dim, reuse=reuse, pretrained_embeddings=pretrained_embeddings, pretrained_embeddings_are_trainable=config['pretrained_embeddings_are_trainable'])
		inp = tf.nn.embedding_lookup(token_emb_mat, token_lookup_sequences_placeholder) 
			
		# run lstm 
		outputs_tensor = self.encoderRNN(inp, lstm_cell_size, config['batch_size'], max_sentence_length , reuse, mode=mode)
		outputs = tf.unstack(outputs_tensor)
		return outputs


	########################################################################
	# DECODER MODEL...

	def attentionLayer(self, encoder_vals, h_prev, sentinel=None, reuse=False): ## pointer network layer
		#print "reuse = ",reuse
		lstm_cell_size = h_prev.get_shape().as_list()[-1]
		with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
			shap = encoder_vals.get_shape().as_list()
			encoder_vals_size = shap[2]
			cell_size = h_prev.shape[1]
			encoder_sequence_length = shap[1]

			## winit = tf.get_variable('winit', [encoder_vals_size, lstm_cell_size] ) # not there in priginal formulation.  needed if concatenating forward and reverse lstms on encoder side. Can be removed if encoder_vals_size=lstm_cell_size
			## encoder_vals = tf.reshape(encoder_vals, [-1,encoder_vals_size])
			## encoder_vals = tf.matmul( encoder_vals, winit )
			## encoder_vals = tf.reshape(encoder_vals, [-1,shap[1],lstm_cell_size]) # N, encoder_sequence_length, lstm_cell_size

			# sentinel : N, lstm_cell_size
			sentinel = tf.get_variable('sentinel_vector',[lstm_cell_size])
			sentinel = tf.expand_dims(sentinel,0)  # 1, lstm_cell_size
			sentinel = tf.tile( sentinel, [shap[0],1] )
			sentinel_expanded = tf.expand_dims(sentinel,1)  # N, 1, lstm_cell_size
			encoder_vals_expanded = tf.concat([sentinel_expanded, encoder_vals], axis=1) # N, encoder_sequence_length+1, lstm_cell_size

			watt = tf.get_variable('watt', [cell_size, cell_size] )
			batt = tf.get_variable('batt', [1, cell_size] )
			query = tf.tanh( tf.matmul(h_prev, watt) + batt ) # . this is "query"
			h_att = tf.expand_dims( query , 1)	# (N,1,cell_size)  
			out_att = tf.reduce_sum( tf.multiply( h_att, encoder_vals_expanded ), axis=2 ) # (N, encoder_sequence_length+1)
			alpha = tf.nn.softmax(out_att)  # (N, encoder_sequence_length+1)
			sentinel_weight = alpha[:,0]
			alpha = alpha[:,1:]
			context = tf.reduce_sum(encoder_vals * tf.expand_dims(alpha, 2), 1, name='context')   #(N, lstm_cell_size)
			#return context, alpha
			return alpha, sentinel_weight, context

	def initgetInitialStateVars(self, encoder_outputs, lstm_cell_size):
		with tf.variable_scope(tf.get_variable_scope(), reuse=None) as scope:
			encoder_avg_output = tf.reduce_mean( encoder_outputs, axis=0) # N,dims
			winit = tf.get_variable('winit', [encoder_avg_output.shape[-1], lstm_cell_size] )
			self.winit = winit
			scope.reuse_variables()

	def getInitialState(self, encoder_outputs, lstm_cell_size, reuse=False):
		with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
			#print "tf.get_variable_scope() = ",tf.get_variable_scope()
			encoder_avg_output = tf.reduce_mean( encoder_outputs, axis=0) # N,dims
			winit = self.winit #tf.get_variable('winit', [encoder_avg_output.shape[-1], lstm_cell_size] )
			encoder_avg_output = tf.matmul( encoder_avg_output, winit )
			decoder_initial_state = [ encoder_avg_output, encoder_avg_output ] # c,h
			return decoder_initial_state

	def runDecoderStep(self, lstm_cell, cur_inputs, prev_cell_output, state, encoder_outputs, sentinel, reuse=False):
		alpha, sentinel_weight, context = self.attentionLayer(encoder_outputs, prev_cell_output, sentinel, reuse) # alpha is (None, encoder_sequence_length)
		#inputs =  cur_inputs 
		inputs = tf.concat([ cur_inputs, context ], axis=1)
		return lstm_cell(inputs, state=state), alpha, sentinel_weight, context


	def initDecoderOutputVariables(self,lstm_cell_size, token_vocab_size):
		with tf.variable_scope('decoder_output', reuse=None) as scope:
			w_out = tf.get_variable('w_out', shape=[lstm_cell_size, token_vocab_size], initializer=tf.random_normal_initializer(-1.0,1.0))
			b_out = tf.get_variable('b_out', shape=[token_vocab_size]) # , initializer=tf.random_normal())
			w_context_out = tf.get_variable('w_context_out', shape=[lstm_cell_size, token_vocab_size], initializer=tf.random_normal_initializer(-1.0,1.0))
			b_context_out = tf.get_variable('b_context_out', initializer=tf.random_normal([token_vocab_size]) )
			scope.reuse_variables()
	def getDecoderOutputVariables(self):
		with tf.variable_scope('decoder_output', reuse=True) as scope:
			w_out = tf.get_variable('w_out')
			b_out = tf.get_variable('b_out')
			w_context_out = tf.get_variable('w_context_out')
			b_context_out = tf.get_variable('b_context_out')
			return w_out, b_out, w_context_out, b_context_out
	
	def getDecoderOutput(self, output, lstm_cell_size, token_vocab_size, out_weights, alpha_sentinel, encoder_input_sequence, batch_size, vocab_size, context, use_context_for_out=config.use_context_for_out): 
		# outputs_list: list of tensor(batch_size, cell_size) with time_steps number of items
		
		#print "out_weights = ",out_weights
		w_out, b_out, w_context_out, b_context_out = out_weights
		alpha, sentinel_weight = alpha_sentinel  #sentinel_weight: N,
		
		pred =  tf.matmul(output, w_out) + b_out  #(N,vocab_size)
		if use_context_for_out:
			pred += (tf.matmul(context, w_context_out) + b_context_out)  #(N,vocab_size)

		pred_softmax = tf.nn.softmax(pred)
		sentinel_weight = tf.expand_dims(sentinel_weight,1) # N,1
		pred = pred_softmax * sentinel_weight  # g * rnnprob(w)
		
		r = tf.expand_dims( tf.range( batch_size ) , 1 )
		encoder_length = tf.shape(encoder_input_sequence)[1]
		r = tf.tile(r,[1,encoder_length]) # batch_size, encoder_length
		r_concat = tf.stack( [r,encoder_input_sequence ], axis=2 ) # batch_size, encoder_length, 2
		r_concat_flattened = tf.reshape(r_concat,[-1,2]) # batch_size * encoder_length, 2
		r_concat_flattened = tf.cast(r_concat_flattened, tf.int64)
		#alpha = alpha * (tf.ones(sentinel_weight.shape) ## sum of alpha is already (1-g) ## - sentinel_weight) 
		# alpha: N,encoder_length. sentinel_weight: N,1
		alpha_flattened = tf.reshape(alpha,[-1]) # batch_size * encoder_length
		alpha_flattened = alpha_flattened   # batch_size * encoder_length
		dense_shape = np.array([batch_size, vocab_size], dtype=np.int64)
		pointer_probs = tf.SparseTensor(indices=r_concat_flattened, values=alpha_flattened, dense_shape=dense_shape)
		pred = tf.sparse_add(pred, pointer_probs)
		
		return pred # Note: these are probabiltiies. use sparse cross entropy with logits only after processing

	def initEmbeddings(self, emb_scope, token_vocab_size, embeddings_dim, pretrained_embeddings_are_trainable=True, reuse=False, pretrained_embeddings=None):
		#print "pretrained_embeddings_are_trainable = ",pretrained_embeddings_are_trainable
		with tf.variable_scope(emb_scope, reuse=reuse):
			if pretrained_embeddings.all()!=None:
				token_emb_mat = tf.get_variable("emb_mat", shape=[token_vocab_size, embeddings_dim], dtype='float', initializer=tf.constant_initializer(np.array(pretrained_embeddings)), trainable=pretrained_embeddings_are_trainable )
				token_emb_mat = tf.concat( [tf.zeros([1, embeddings_dim]), tf.slice(token_emb_mat, [1,0],[-1,-1]) ], axis=0 )	
			else:
				token_emb_mat = tf.get_variable("emb_mat", shape=[token_vocab_size, embeddings_dim], dtype='float')
				# 0-mask
				token_emb_mat = tf.concat( [tf.zeros([1, embeddings_dim]), tf.slice(token_emb_mat, [1,0],[-1,-1]) ], axis=0 )	
				#print "token_emb_mat = ",token_emb_mat
		return token_emb_mat

	def greedyInferenceModel(self, params ):
		lstm_cell = params['lstm_cell']
		token_vocab_size = params['vocab_size']
		lstm_cell_size = params['lstm_cell_size']
		batch_size = params['batch_size']
		embeddings_dim = params['embeddings_dim']
		batch_time_steps = params['max_output_seq_length']
		token_emb_mat = params['token_emb_mat']
		out_weights = params['output_vars']
		encoder_outputs = params['encoder_outputs']
		cell_output, state = params['cell_state']
		encoder_input_sequence = params['encoder_input_sequence']
		sentinel = params['sentinel']

		num_steps = batch_time_steps
		outputs = []
		alpha_all = []

		for time_step in range(num_steps):
			if time_step==0:
				inp = tf.ones([batch_size,1], dtype=tf.int32) #start symbol index  #TO DO: get start index from config
				#outputs.append( tf.reshape(inp,[batch_size]) )
			inputs_current_time_step = tf.reshape( tf.nn.embedding_lookup(token_emb_mat, inp) , [-1, embeddings_dim] )
			if time_step > 0: tf.get_variable_scope().reuse_variables()
			
			(cell_output, state), alpha_cur, sentinel_weight, context = self.runDecoderStep(lstm_cell=lstm_cell, cur_inputs=inputs_current_time_step, encoder_outputs=encoder_outputs, prev_cell_output=cell_output, sentinel=sentinel, reuse=(time_step!=0), state=state)
			cur_outputs = self.getDecoderOutput(cell_output, lstm_cell_size, token_vocab_size, out_weights, (alpha_cur,sentinel_weight), encoder_input_sequence, batch_size, token_vocab_size, context )
			assert cur_outputs.shape[1]==token_vocab_size
			word_predictions = tf.argmax(cur_outputs,axis=1)
			outputs.append(word_predictions)
			alpha_all.append(alpha_cur)
			inp = word_predictions
		return outputs, alpha_all


	def decoderRNN(self, x, params, reuse=False, mode='training'):

		lstm_cell = params['lstm_cell']
		encoder_outputs = params['encoder_outputs']
		token_vocab_size = params['vocab_size']
		lstm_cell_size = params['lstm_cell_size']
		batch_size = params['batch_size']
		embeddings_dim = params['embeddings_dim']
		batch_time_steps = params['max_output_seq_length']
		encoder_input_sequence = params['encoder_input_sequence']

		if 'token_emb_mat' in params:
			token_emb_mat = params['token_emb_mat']
		else:
			token_emb_mat = None

		#with tf.variable_scope('decoder'):
		num_steps = None
		if batch_time_steps:
			num_steps = batch_time_steps #should be equal to x.shape[1]
		else:
			#print "x.shape: ",x.shape
			num_steps = x.shape[1]
		inputs = x

		self.decoder_cell = cell = lstm_cell

		# inital state
		self.initgetInitialStateVars(encoder_outputs, lstm_cell_size)
		decoder_initial_state = self.getInitialState(encoder_outputs, lstm_cell_size, reuse=reuse)

		#decoder output variable
		self.initDecoderOutputVariables(lstm_cell_size,token_vocab_size)
		out_weights = self.getDecoderOutputVariables()
		#w_out, b_out, w_context_out, b_context_out = out_weights

		#unrolled lstm 
		outputs = [] # h values at each time step
		vals = []
		state = decoder_initial_state
		cell_output = state[1]
		encoder_outputs = tf.stack(encoder_outputs) # timesteps, N, cellsize
		encoder_outputs = tf.transpose(encoder_outputs,[1,0,2]) # N, timesteps, cellsize 
		sentinel = None #tf.ones([batch_size,lstm_cell_size], dtype=tf.float32) # Not used

		with tf.variable_scope("RNN"):
			if mode=='training':
				eps = tf.constant(0.000000001, dtype=tf.float32)
				sentinel_loss = []
				decoder_output_inpmatch_sequence = params['decoder_output_inpmatch_sequence']
				pred = []
				for time_step in range(num_steps):
					if time_step > 0: tf.get_variable_scope().reuse_variables()
					inputs_current_time_step = inputs[:, time_step, :]
					(cell_output, state), alpha, sentinel_weight, context = self.runDecoderStep(lstm_cell=lstm_cell, cur_inputs=inputs_current_time_step, encoder_outputs=encoder_outputs, prev_cell_output=cell_output, sentinel=sentinel, reuse=(time_step!=0), state=state)
					cur_pred = self.getDecoderOutput(cell_output, lstm_cell_size, token_vocab_size, out_weights, (alpha,sentinel_weight), encoder_input_sequence, batch_size, token_vocab_size, context)
					pred.append(cur_pred)

					cur_decoder_output_inpmatch_sequence = decoder_output_inpmatch_sequence[:, time_step, :] # N,inp_seq_length 
					# alpha: N, inp_seq_length
					cur_sentinel_attention_loss = tf.reduce_sum( alpha * cur_decoder_output_inpmatch_sequence, axis=1 ) # N
					sentinel_loss.append(  -tf.log(sentinel_weight+ cur_sentinel_attention_loss) ) 
					#vals.append([cur_sentinel_attention_loss,sentinel_weight])

				pred = tf.stack(pred), sentinel_loss # sentinel_loss: T, N
				tf.get_variable_scope().reuse_variables()
				#self.vals = vals

			elif mode=='inference':

				#Greedy
				params['output_vars'] = out_weights
				params['cell_output'] = cell_output
				params['encoder_outputs'] = encoder_outputs
				params['cell_state'] = cell_output, state
				params['beam_size'] = 20
				params['sentinel'] = sentinel
				outputs, alpha =  self.greedyInferenceModel(params) #self.beamSearchInference(params)  #self.greedyInferenceModel(params)
				ret_encoder_outputs = tf.transpose(encoder_outputs,[1,0,2]) # N, timesteps, cellsize 
				pred = outputs, ret_encoder_outputs, alpha

		return pred


	#################################################################################################################


	def getDecoderModel(self, config, encoder_outputs, is_training=False, mode='training', reuse=False, bucket_num=0 ):

		token_vocab_size = config['vocab_size']
		max_sentence_length = config['max_output_seq_length']
		embeddings_dim = config['embeddings_dim']
		lstm_cell_size = config['lstm_cell_size']
		if mode=="inference":
			pass

		#placeholders
		if mode=='training':
			token_lookup_sequences_decoder_placeholder =self.token_lookup_sequences_decoder_placeholder_list[bucket_num]
			masker = self.masker_list[bucket_num]
			token_output_sequences_placeholder = self.token_output_sequences_decoder_placeholder_list[bucket_num]
			encoder_input_sequence = self.token_lookup_sequences_placeholder_list[bucket_num] # encoder
			decoder_output_inpmatch_sequence = self.token_output_sequences_decoder_inpmatch_placeholder_list[bucket_num]
		else:
			encoder_input_sequence = self.token_lookup_sequences_placeholder_inference

		#embeddings
		share_embeddings=config['share_encoder_decoder_embeddings']
		emb_scope = 'emb_decoder'
		if reuse:
			token_emb_mat = self.decoder_token_emb_mat
		else:
			pretrained_embeddings=None
			if share_embeddings:
				# ignoreing pretrained embeddings if shared embeddings
				self.decoder_token_emb_mat = token_emb_mat = self.encoder_token_emb_mat
			else:
				if config['pretrained_embeddings']:
					pretrained_embeddings = config['decoder_embeddings_matrix']
				self.decoder_token_emb_mat = token_emb_mat = self.initEmbeddings(emb_scope, token_vocab_size, embeddings_dim, reuse=reuse, pretrained_embeddings=pretrained_embeddings, pretrained_embeddings_are_trainable=config['pretrained_embeddings_are_trainable'])

		with tf.variable_scope('decoder',reuse=reuse):
				
			# lstm 
			lstm_cell = rnn.BasicLSTMCell(lstm_cell_size, forget_bias=1.0, state_is_tuple=True, reuse=reuse)

			if mode=='inference':
				params={k:v for k,v in config.items()}
				params['lstm_cell'] = lstm_cell 
				self.decoder_lstm_cell = lstm_cell
				params['encoder_outputs'] = encoder_outputs
				params['token_emb_mat'] = token_emb_mat
				params['encoder_input_sequence'] = encoder_input_sequence
				inp= None #tf.nn.embedding_lookup(token_emb_mat, token_lookup_sequences_decoder_placeholder)
				pred = self.decoderRNN(inp, params, mode='inference')
			elif mode=='training':
				params={k:v for k,v in config.items()}
				params['lstm_cell'] = lstm_cell 
				params['encoder_input_sequence'] = encoder_input_sequence
				params['encoder_outputs'] = encoder_outputs
				params['decoder_output_inpmatch_sequence'] = decoder_output_inpmatch_sequence

				#params['token_emb_mat'] = None
				inp = tf.nn.embedding_lookup(token_emb_mat, token_lookup_sequences_decoder_placeholder) 
				pred, sentinel_loss = self.decoderRNN(inp, params, mode='training')  # timesteps, N, vocab_size
				pred_for_loss = tf.log(pred) # since sparse_softmax_cross_entropy_with_logits takes softmax on its own as well
				pred = tf.unstack(pred)
				#pred = tf.stack( [ tf.nn.softmax(vals) for vals in pred ] )  ## pred is already a prob distribution

				if is_training:
					pred_masked = pred_for_loss 
					pred_masked = tf.transpose( pred_masked , [1,0,2] ) # N, timesteps, vocabsize
					cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_masked, labels=token_output_sequences_placeholder) # token_output_sequences_placeholder is N,timesteps. cost will be N, timesteps
					cost = tf.multiply(cost, masker)  # both masker and cost is N,timesteps. 
					cost = tf.reduce_sum(cost) # N
					sentinel_loss = tf.transpose( tf.stack(sentinel_loss) ) 
					sentinel_loss = tf.multiply(sentinel_loss, masker)  # both masker and sentinel_loss is N,timesteps. 
					sentinel_loss = tf.reduce_sum(sentinel_loss) # N
					masker_sum = tf.reduce_sum(masker) # N
					cost = tf.divide(cost, masker_sum) # N
					sentinel_loss = tf.divide(sentinel_loss, masker_sum) # N
					self.sentinel_loss = sentinel_loss 
					if params['use_sentinel_loss']:
						self.cost = cost + params['lambd'] * sentinel_loss
					else:
						self.cost = cost

			return pred #[ tf.nn.softmax(vals) for vals in pred]

	
