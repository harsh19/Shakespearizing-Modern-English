data_dir = "../../data/"

# preprocessing params
max_input_seq_length = 25
max_output_seq_length = 25
#dropout_val = 0.2
do_vocab_pruning = True
max_vocab_size = 12000

# Pointer or seq2seq
use_pointer=True 

# model config
lstm_cell_size=192
embeddings_dim = 192
use_reverse_encoder=True
share_encoder_decoder_embeddings=True
use_pretrained_embeddings = True
pretrained_embeddings_path = data_dir + "embeddings/retrofitted_external_192_startend.p"
pretrained_embeddings_are_trainable = False  
use_additional_info_from_pretrained_embeddings = True # if some word is not in training data set but is there in pretrained embeddings: mark True to add such words also. Otherwise mark False

# Specific to pointer model
use_sentinel_loss = False
lambd = 2.0
use_context_for_out = True

# general training params
display_step=1
sample_step=2
save_step = 1
batch_size = 32
