# -*- coding: utf-8 -*-
"""Working  Getting started with TensorFlow_Keras for Machine Translation

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QarVk0HIJfLNKmVZH0SJmaLypBNWBMrQ
"""

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard

# import dependencies
import os
import tensorflow as tf
import datetime
from tensorflow.keras.callbacks import TensorBoard
import time
import Constant
import manipulateFolder 
import zipfile
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, History, CSVLogger
import numpy as np
NAME =""

"""# Data"""


modern = 'data/train.modern.nltktok'
original = 'data/train.original.nltktok'

"""# Extract vocabulary"""

"""## Modern English"""


# read text
with open(modern) as file:
  text = file.read()

# tokenize
text = text.lower()
text = text.split('\n')
tokens = []
for sent in text:
  tokens.extend(sent.split(' '))

# get vocabs
source_vocabs = ['[UNK]', '[PAD]'] + list(set(tokens))

print("Vocabulary size = {}".format(len(source_vocabs)))

"""## Shakespearse's English"""

# read text
with open(original) as file:
  text = file.read()

# tokenize
text = text.lower()
text = text.split('\n')
tokens = []
for sent in text:
  tokens.extend(sent.split(' '))

# get vocabs
target_vocabs = ['[UNK]', '[PAD]'] + list(set(tokens))

print("Vocabulary size = {}".format(len(target_vocabs)))

target_vocabs[0]

"""## Load data"""

def load_data(modern, original, source_vocabs, target_vocabs):
    #  read data
    modern = tf.data.TextLineDataset(modern)
    original = tf.data.TextLineDataset(original)

    # batching
    batch_size = Constant.BATCH_SIZE # multiple of 2: 2, 4, 8, 16, 32, 64
    modern = modern.batch(batch_size)
    original = original.batch(batch_size)

    ## lower case
    lower_case = lambda x: tf.strings.lower(x)
    modern = modern.map(lower_case)
    original = original.map(lower_case)

    ## tokenize
    split = lambda x: tf.strings.split(x)
    modern = modern.map(split)
    original = original.map(split)

    ## remove stopwords/punctuations
    ## please explore the list of stopword and punctuations and remove them
    ## use tf.strings.regex_replace to remove all stopword and puncutations
    ## your codes go here

    ## padding & truncation 
    def pad(text):
        """
        text : RaggedTensor (list of variable-length elements)
        hence, use tf.map_fn
        """
        max_length = 50
        def _pad(input):
            if tf.size(input) < max_length:
                paddings = tf.repeat(tf.constant('[PAD]', dtype = tf.string), max_length - tf.size(input))
                return tf.concat([input, paddings], axis = 0)
            else:
                return input[:max_length]
            
        # map _pad for each string
        text = tf.map_fn(_pad, text)

        return text.to_tensor() # text is tf.RaggedTensor for dynamic length -> converted to fixed length
    # padding
    modern = modern.map(pad)
    original = original.map(pad)

    # encode into integers
    modern_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            source_vocabs, tf.constant(list(range(len(source_vocabs))))
        ), default_value = 0
    )
    
    original_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            target_vocabs, tf.constant(list(range(len(target_vocabs))))
        ), default_value = 0
    )
    
    modern = modern.map(lambda text: modern_table.lookup(text))
    original = original.map(lambda text: original_table.lookup(text))

    dataset = tf.data.Dataset.zip((modern, original))
    return dataset

train_dataset = load_data(modern, original, source_vocabs, target_vocabs)

for sample in train_dataset:
  print('modern', sample[0])
  print('original', sample[1].shape)
  break

val_modern = 'data/valid.modern.nltktok'
val_original = 'data/valid.original.nltktok'

val_dataset = load_data(val_modern, val_original, source_vocabs, target_vocabs)


"""# Model and Tensorboard"""

def model_fn(source_vocab_size, target_vocab_size, sequence_length):
  # input
  input = tf.keras.Input(shape = (sequence_length))

  # embedding layer
  embedding = tf.keras.layers.Embedding(source_vocab_size, 300) # vocabs = 10000, embed_dim = 64, sequence_length = 10
  embedding = embedding(input)

  # LSTM encoder
  lstm = tf.keras.layers.LSTM(256, return_sequences = True, name = 'Encoder_1',activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.0,unroll=False,use_bias=True)(embedding)
  lstm = tf.keras.layers.LSTM(512, return_sequences = True, name = 'Encoder_2',activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.0,unroll=False,use_bias=True)(lstm)

  # LSTM decoder
  lstm = tf.keras.layers.LSTM(128, return_sequences = True, name = 'Decoder',activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.0,unroll=False,use_bias=True)(lstm)

  # output
  output = tf.keras.layers.Dense(target_vocab_size)(lstm)

  return tf.keras.Model(inputs = [input], outputs = [output])

model = model_fn(len(source_vocabs), len(target_vocabs), sequence_length = 50)
print(model.summary())

"""Naming the Model"""


def name_model(epochs,type_of_model,learning_rate,loss_function):
    return f' model is {type_of_model} the epochs {epochs}  learning_rate {learning_rate} ' \
           f'lost function is {loss_function}'
#Early stop in keras
class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


"""# Hyperparameters"""

learning_rate = Constant.LEARNING_RATE # 0.0001, 0.00xxx1
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
metrics = [] # BLEU
model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
"""Nameing the tensorboad
"""
type_of_model = "seq_seq"
loss_function = "Sparse_Categorigical_Crossentropy"
NAME = name_model(Constant.EPOCHS,type_of_model,learning_rate,loss_function)
NAME = NAME+' '+format(datetime.datetime.now())
NAME = NAME.replace(":","_")
csv_Name = NAME+".log"
"""# Tranining

"""
history = History()
logs = Callback()
csv_logger = CSVLogger(csv_Name)
tensorboard  = TensorBoard(log_dir="logs/{}".format(NAME))
epochs = Constant.EPOCHS
model.fit(train_dataset, validation_data = val_dataset, epochs = epochs, verbose = 1, callbacks=[history,csv_logger,tensorboard,EarlyStoppingAtMinLoss()])


"""## Save model"""

size = len(history.history['loss'])
model_loss = history.history['loss'][size-1]
model_loss_formated = format(model_loss,".5f")
model_validation_loss = history.history['val_loss'][size-1]
model_validation_loss_formated = format(model_validation_loss,".5")
name_of_model = NAME
location_of_folder = "/model/"
model.save(location_of_folder+name_of_model)

#move the csv file to the

manipulateFolder.moveFileIntoDir(csv_Name,"csv")


'''model = tf.keras.load_model(path)
model.predict(input)'''
print(name_of_model)

