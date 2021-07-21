# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import collections
import logging
import os
import re
import string
import sys
import time

import numpy as np

import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import dtype
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

# %% [markdown]
# The dataset that will be imported is a dictionary containing the trainning and validation sets

# %%
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']


# %%
for pt_examples, en_examples in train_examples.batch(3).take(1):
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'))
    print()
    
    for en in en_examples.numpy():
        print(en.decode('utf-8'))
    


# %%



# %%
train_en = train_examples.map(lambda pt, en: en ) #method map is provided by tensorflow
train_pt = train_examples.map(lambda pt, en: pt )


# %%



# %%
type(train_examples)


# %%
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab


# %%
bert_tokenizer_params = dict(lower_case = True)


# %%
reserved_tokens = ['[PAD]', '[UNK]', '[START]', '[END]']

bert_vocab_args = dict(vocab_size = 8000,
                       reserved_tokens = reserved_tokens,
                       bert_tokenizer_params=bert_tokenizer_params, 
                       learn_params={})


# %%
pt_vocab = bert_vocab.bert_vocab_from_dataset(train_pt.batch(1000).prefetch(2), **bert_vocab_args)


# %%
print(pt_vocab[:10])
print(pt_vocab[100: 110])
print(pt_vocab[1000:1010])
print(pt_vocab[-10:])


# %%
def write_vocab_file(filepath, vocab):
    with open(filepath, 'w', encoding='utf-8') as f:
        for token in  vocab:
            print(token, file=f)


# %%
write_vocab_file("pt_vocab.txt", pt_vocab)


# %%
en_vocab = bert_vocab.bert_vocab_from_dataset(train_en.batch(1000).prefetch(2), **bert_vocab_args)


# %%
write_vocab_file("en_vocab.txt", en_vocab)

# %% [markdown]
# ## Build the tokenizer
# After having the dictionary we have to tokenize our dataset:
# - Divide the dataset into sentences, where each word in a sentence is one-hot vector encoding, taken from the vocabulary

# %%
pt_tokenizer = text.BertTokenizer('pt_vocab.txt', **bert_tokenizer_params)
en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)


# %%
for pt_examples, en_examples in train_examples.batch(3).take(1):
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'))
    print()
    for en in en_examples.numpy():
        print(en.decode('utf-8'))


# %%
examples = train_examples.batch(1).take(3)
examples_numpy = list(examples.as_numpy_iterator())

# %% [markdown]
# "examples" is a list of tuples [(array_pt([]), array_en([])), (), ()], in this case a list with 3 elements:
#   

# %%
type(examples_numpy[0])
examples_numpy[0]


# %%
examples_numpy


# %%
pt_examples = []
en_examples = []
for example in examples_numpy:
    
    pt_examples.append(example[0])
    en_examples.append(example[1])

    


# %%
print(pt_examples)
print(en_examples)


# %%
token_batch = pt_tokenizer.tokenize(pt_examples)
token_batch


# %%
token_batch = token_batch.merge_dims(-2, -1)


# %%
for ex in token_batch.to_list():
    print(ex)


# %%
txt_tokens = tf.gather(pt_vocab, token_batch)
txt_tokens


# %%
tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1)

# %% [markdown]
# ### We have seen how to transform words into a list o indexes, but we now need an embedding for each index

# %%
model_name = "ted_hrlr_translate_pt_en_converter"


# %%
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir='.', cache_subdir='', extract=True
)


# %%
tokenizers = tf.saved_model.load(model_name)


# %%
#Show hall methods available for tokenizers
#The same methods are available in both language tokenizers

[item for item in dir(tokenizers.en) if not item.startswith('_')]


# %%
for pt, en in examples:

    encoded_examples_pt = tokenizers.pt.tokenize(pt)

    print(encoded_examples_pt.to_list())
    


# %%
encoded_examples_pt.to_list()


# %%
def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    pt = pt.to_tensor()
    
    en = tokenizers.en.tokenize(en)
    
    en = en.to_tensor()
    
    return pt, en


# %%
BUFFER_SIZE = 20000
BATCH_SIZE = 64


# %%
def make_batches(ds):
  return (ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))


# %%
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)


# %%
test = train_batches.take(1)
print(len(list(test.as_numpy_iterator())[0][0])) # it takes out 64 padded sentences


# %%
print(list(test.as_numpy_iterator()))


# %%
print(list(test.as_numpy_iterator())[0][0]) #list[tuple(t1, t2)]


# %%
print(type(list(test.as_numpy_iterator())[0][0])) #list[tuple(array-1, array-2)] # 64 pt padded  sentences, and 64 en padded sentences


# %%
def get_angles (pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) /np.float32(d_model))
    return pos * angle_rates


# %%
# We will create all positional econdings for all positions:
# for example:
# if our sentences have e fixed lenth of "l"
# an embedding size of "d"
# with the next method The result will be matrix with the size
# (l, size) Therefore along the the different rows we will have the positional encoding for the different positions 


# %%
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) # starting from zero and on every 2 elements replace
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) # starting from zero and on every 2 elements replace

    pos_encoding = angle_rads[np.newaxis, ...] # elipsis is just to continue the rest o the array, np.newaxis adds a new axis, important when we are trying to broadcast an array
    
    return tf.cast(pos_encoding, dtype=tf.float32) 


# %%
n, d = 2048, 512
pos_encoding = positional_encoding(n, d)


# %%
pos_encoding.shape


# %%
def create_padding_mask(seq):
    # From a squence of token indexes [12, 4, 0, 10, 50], if the index is 0 we will create a mask of the type [0, 0, 1, 0, 0]
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    return seq [:, tf.newaxis, tf.newaxis, : ] # (batch_size, 1, 1, seq_len)
    # why do we add these new axis ? (comeback here after)


# %%
x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
mask = create_padding_mask(x)


# %%
weights = tf.nn.softmax(tf.ones((3, 5, 5)) + mask*(-1000), -1)


v = tf.ones((3, 5, 10))

result = tf.matmul(weights, v)

result.shape


# %%
tf.linalg.band_part(tf.ones((4, 5)), -1, 0)


# %%
def create_look_ahead_mask(size):
    """ If your ouput is of size (1, s), where s in the size of sentence, then you will need to feed the network
    first with only the first element, so that you predict the second, after you will need to feed the 1st and 2nd, an so on
    
    You will need a matrix of size (s, s) where: 
    
    [1, 0, 0, 0]
    [1, 1, 0, 0]
    [1, 1, 1, 0]
    [1, 1, 1, 1]
    because mask is to be applied on the attention layer before scalar softmax is computed, by the process of [scalar product matrix]  + mask * -large number
    
    then the matrix will have to be of the form
    
    [0, 1, 1, 1]
    [0, 0, 1, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 0]
    
    This is so because in softmax e comput e-z if z is very large number then e-z will be close to 0, meaning that we will add a large negative number to places where
    the mask is 1 and 0 to places where the mask is 0  
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    
    return mask
    


# %%
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions,
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_lenv.
    The mask has different shapes depending on its type (padding or lookahead)
    but it must be broadcastable for addition
    
    Args:
        
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth) depth is embbedding size
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k) defaults  to None
        
    Returns:
        output, attention weights
        
     """

    matmul_qk = tf.matmul(q, k, transpose_b=True) #(..., seq_len_q, seq_len_k)
     
     #scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32) # gets the depth of k, by returning the shape of k and then transforming it tensor of type float32
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
     

    #Add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    #softmax is normalized  on the last axis (seq_len_k) so that the scores add up to 1, i.e.: all the columns add up to 1
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)
    
    output = tf.matmul(attention_weights, v) # (..., seq_len_k, depth_v)
    return output, attention_weights 
        


# %%
def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print("Attention weights are :")
    print(temp_attn)
    print("Output is: ")
    print(temp_out)


# %%
"TEST"
np.printoptions(supress=True)

temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=tf.float32)  # (4, 3)


temp_v = tf.constant([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype=tf.float32)  # (4, 2)

temp_q = tf.constant([[0, 0, 10],
                      [0, 10, 0],
                      [10, 10, 0]], dtype=tf.float32)  # (3, 3)

print_out(temp_q, temp_k, temp_v)


# %%
class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads):
        
        super(MultiHeadAttention, self).__init__() # init tf.keras.layer
        
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads # We will set (d_model) to the total size of the ouput transformed embbedings, but we will divide this vector into 
        #different heads

        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        
        """Split the last dimension into (num_heads, depth)
        Transpose  the result  such that  the shape  is (batch_size, num_heads, seq_len, depth)
        
        """

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3]) # permutation argument is a position that we want the original axis to take in the new matrix
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q) # (batch_size, seq_len, d_model) # On the forward pass the word embbedings will go through this 3 different Dense layers (Batch_Size, Seq_len, depth of embbeding)
        k = self.wk(k) # (batch_size, seq_len, d_model) # It means that each embbeding vector with depth d will go through a 3 Dense Layers, this dense layer will have d input units
        v = self.wv(v) # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)


        # scaled_attention.shape  == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights == (batch_size, num_heads, seq_len_q, seq_len_k)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # (batch_size, seq_len_q, d_model)
                                      
        
        output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)

        return output,  attention_weights


# %%
temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 70))

out, attn = temp_mha(y, k=y, q=y, mask=None)

out.shape, attn.shape

# %% [markdown]
# ## Point Wise feed Forward Network

# %%

def point_wise_feed_forward_network(d_model, dff):
    """
    Feed forward Network with 2 fully connected layers, with sizes dff and d_model respectively
    
    """

    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(d_model)])


# %%
sample_ffn = point_wise_feed_forward_network(512, 2048)
sample_ffn(tf.random.uniform((64, 50, 512))).shape

# %% [markdown]
# ## Encoder Layer

# %%
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        
        """
        Arguments:
        d_model--[int], depth of the concatenated output to be passed through the decoder
        num_heads--[int], number of heads; the sum of the vector depth of each head will be equal to d_model, d_model/ num_heads = depth of each transformed emmbedding, at each head
        dff--[int], number of units for the first layer of the point wise feed forward network

        Returns:
        out2--[tf tensor](bacth_size, input_seq_len, d_model) to be passed throuhg the decoder
        
        """
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask) # (batch_size, input_seq_len, d_model
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) # skip  connection, (batch_size, input_seq_len, d_model

        
        
        ffn_output = self.ffn(out1)# (batch_size, input_seq_len, d_model
        ffn_output = self.dropout2 (ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model

        return out2


        
        
        


# %%
sample_encoder_layer = EncoderLayer(512, 8, 2048)
sample_encoder_layer_output = sample_encoder_layer(
    tf.random.uniform((64, 43, 512)), False, None)# No training, no mask

sample_encoder_layer_output.shape

# %% [markdown]
# ## Decoder Layer

# %%
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        
        #enc_outpout.shape == (batch_size, input_seq_len, d_model)
        
        attn1,  attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask) # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask) # (batch_size, target_seq_len, d_model)

        attn2 = self.dropout2(attn2, training)
        out2 = self.layernorm2(attn2 + out1)# (batch_size, target_seq_len, d_model)

        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)#(batch_size, target_seq_len, d_model)
        
        return out3, attn_weights_block1, attn_weights_block2




# %%
sample_decoder_layer = DecoderLayer(512, 8, 2048)
sample_decoder_layer_output, _, _ = sample_decoder_layer(tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, False, None, None)
sample_decoder_layer_output.shape #(batch_size, targe_seq_len, d_model)

# %% [markdown]
# ## Decoder

# %%
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model) # 1, seq_len, d_model
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        """
        Arguments
        x -- tensor (batch_size, seq_len) [[1, 3,1800, 3493, ..., 1234 ], [.....]]
        """

        seq_len = tf.shape(x)[1]
        x = self.embedding(x) #batch_size, seq_len, d_model
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :] #limit positional encoding to seq_len
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            
        
        return x # (batch_size, input_seq_len, d_model)


# %%
sample_encoder = Encoder(num_layers=2, 
                         d_model=512, 
                         num_heads=8,
                         dff=2048, 
                         input_vocab_size=8500,
                         maximum_position_encoding=10000)

temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200) #(batch_size=64, seq_len=62)

sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

print(sample_encoder_output.shape) # (batch_size, input_seq_len, d_model)


# %%
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        
        seq_len = tf.shape(x)[1]
        
        attention_weights = {} # we know have two attentions weights per layer, one from the output attention and another, with encoder output + decoder first attention output
        
        x = self.embedding(x)# (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x += self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask) 
            
            attention_weights[f"decoder_layer{i + 1}_block1"] = block1
            attention_weights[f"decoder_layer{i + 1}_block2"] = block2
            
        return x, attention_weights


# %%
sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, 
                         dff=2048, target_vocab_size=8000, 
                         maximum_position_encoding=5000)

temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)


output, attn = sample_decoder(temp_input,
                              enc_output=sample_encoder_output,
                              training=False, 
                              look_ahead_mask=None,
                              padding_mask=None )
output.shape, attn['decoder_layer2_block2'].shape

# output [batch_size, sequece_len, depth]
# attn [batch_size, num_heads, seq_len, ]


# %%
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        
        super(Transformer, self).__init__()
        
        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        
        enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights



# %%

sample_transformer = Transformer(num_layers=2, 
                                 d_model=512, 
                                 num_heads=8,
                                 dff=2048,
                                 input_vocab_size=8500,
                                 target_vocab_size=8000,
                                 pe_input=10000,
                                 pe_target=6000)

temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                               enc_padding_mask=None,
                               look_ahead_mask=None,
                               dec_padding_mask=None)

fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)


# %% [markdown]
## Set hyperparameters

#%%

num_layers = 4
d_model = 128
dff = 512
num_heads  = 8
dropout_rate = 0.1

#%%
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# %%
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


# %%
temp_learning_rate_schedule = CustomSchedule(d_model)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')


# %%[markdown]
### Loss and Metrics


# %%
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# %%
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0)) #if we output index 0 which the padding representation we want to exclude this word from the loss function
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def accuracy_function(real, pred):
    
    accuracies = tf.equal(real, tf.argmax(pred, axis=2)) # why do we have axis = 2
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)    
    
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask) 
    
    
#%%
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

# %%
transformer = Transformer(num_layers=num_layers,
                          d_model=d_model,
                          num_heads=num_heads,
                          dff=dff, 
                          input_vocab_size=tokenizers.pt.get_vocab_size(),
                          target_vocab_size=tokenizers.en.get_vocab_size(),
                          pe_input=1000,
                          pe_target=1000,
                          rate=dropout_rate)
#%%
def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    
    # Used in the 2nd attention block in the decoder
    # This padding mask is used to mask the encoder outputs 
    dec_padding_mask = create_padding_mask(inp)
    
    #Used  in the 1st attention block in the decoder
    #It is used to pad and mask future tokens in the input received  by the decoder
    
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask
    
#%%

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a chekpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# %%
EPOCHS = 20 


# %%
train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64), 
                        tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
#%%
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    
    with tf.GradientTape() as tape:
        
        predictions, _  = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        
        loss = loss_function(tar_real, predictions)
        
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        
        train_accuracy(accuracy_function(tar_real, predictions))
# %%
for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    #inp ->portuguese, tar->english
    
    for (batch, (inp, tar)) in enumerate(train_batches):
        
        train_step(inp, tar)
        
        if batch % 50 == 0 :
            print(f"Epoch {epoch + 1} Batch {batch} Loss {train_loss.result()} Accuracy {train_accuracy.result()}")

    if (epoch + 1) % 5 == 0:
        
        ckpt_save_path = ckpt.save()
        
        print(f"Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}")

    print(f"Epoch {epoch + 1} Loss {train_loss.result()} Accuracy {train_accuracy.result()}")
    print(f"Time taken for 1 epoch: {time.time() - start } secs \n")
    
#%%
example_iter = iter(train_batches)
#%%

def evaluate (sentence, max_length=40):
    # inp sentence is portuguese, hence adding the start and en token
    sentence = tf.convert_to_tensor([sentence]) 
    sentence = tokenizers.pt.tokenize(sentence).to_tensor() # This adds the start and end token
    
    encoder_input = sentence

    # as the target is english, the first word to  the transformer should be  the english sentence
    start, end = tokenizers.en.tokenize([''])[0] # tokenizin and empty string will result on adding start token followed be the end token
    output = tf.convert_to_tensor([start])# we will feed the decoder with the start token only
    output = tf.expand_dims(output, 0) #adds one more dimension 
    
    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        
        # predictions.shape == (batch_size, seq_len, target_vocab_size)
        
        predictions, attention_weights =  transformer(encoder_input,
                                                      output,
                                                      False,
                                                      enc_padding_mask, 
                                                      combined_mask,
                                                      dec_padding_mask)
        
        # select the last word from the seq_len dimension
        
        predictions = predictions[:, -1, :] # (batch_size, 1, vocab_size)
        
        predicted_id = tf.argmax(predictions, axis=-1)
        
        #concatenate the predicted_id to the output which is given to the decoder as its input
        output = tf.concat([output, predicted_id], axis=-1)
        
        #return the result if the predicted_id is equal to the end  token
        if predicted_id == end:
            break
    
        #output.shape(1, tokens)
        
        text = tokenizers.en.detokenize(output)[0] # generates english readable text
        
        tokens = tokenizers.en.lookup(output)[0] # generate a list with the tokens, may not be easily readable like : ##ro
        
        return text, tokens, attention_weights

#%%
def print_translation(sentence, tokens, ground_truth):
    
    print(f"{'Input':15s}: {sentence}")
    print(f"{'Prediction':15s}: {tokens.numpy().decode('utf-8')}")
    print(f"{'Ground truth':15s}:{ground_truth}")
#%%
def plot_attention_head(in_tokens, translated_tokens, attention):
    # the plot is of the attention when a token was generated
    # The model didn't generate <Start> in the output.
    
    translated_tokens = translated_tokens[1:]
    
    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))
    
    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(labels, rotation=90)
    
    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    
    ax.set_yticklabels(labels)
    
#%%

def plot_attention_weights(sentence, translated_tokens, attention_heads):
    
    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.pt.lookup(in_tokens)[0]
    
    fig = plt.figure(figsize=(16, 8))
    
    for h, head in enumerate(attention_heads):
        
        ax = fig.add_subplot(2, 4, h+1)
        
        plot_attention_head(in_tokens, translated_tokens, head)
        
        ax.set_xlabel(f'Head {h+1}')
    
    plt.tight_layout()
    plt.show()

