#Eric Hsieh 2020.09.15
#using huggingface_transformers https://github.com/huggingface/transformers
#using chinese_bert_wwm https://github.com/ymcui/Chinese-BERT-wwm

import tensorflow as tf
import tensorflow_datasets
from transformers import *


TAKE_SIZE = 500
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EPOCHS_NUM = 20
CSV_PATH = "/home/csliao/tf01/dataset/adclu2nmqgk82R.csv"


tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
model = BertModel.from_pretrained("hfl/chinese-bert-wwm")

import pandas as pd
def load_newsdata():
    # load raw source data
    RAW_CSV = CSV_PATH
    df = pd.read_csv(RAW_CSV, sep=',', encoding='utf8')
    return(df)

df = load_newsdata()
df1 = df[['title', 'content', 'labeled']]
#print(df1)

#encoding
def encode_words(s):
  tokens = tokenizer.tokenize(s)
  tokens.append('[SEP]')
  return tokenizer.convert_tokens_to_ids(tokens)
  

titles_list = tf.ragged.constant([encode_words(titles) for titles in df1['title'].values])
contents_list = tf.ragged.constant([encode_words(contents) for contents in df1['content'].values])

#creating tensors
cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*titles_list.shape[0]
titles_list = tf.concat([cls, titles_list], axis=-1)
contents_list = tf.concat([cls, contents_list], axis=-1)
labels_list = tf.ragged.constant(df1['labeled'].values)

#for this content only files ... 
content_tensor = contents_list.to_tensor()
train_data_set = tf.data.Dataset.from_tensor_slices((content_tensor,labels_list))

#splits into two data?

train_data = train_data_set.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = train_data_set.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)

#check by entering tokenizer.vocab_size
vocab_size = 21128

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size,64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
for units in [64,1024]:
  model.add(tf.keras.layers.Dense(units, activation = 'relu'))
model.add(tf.keras.layers.Dense(2))

opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data, epochs=EPOCHS_NUM , validation_data=test_data)


###evaluate loss and accuary for the test_data)
eval_loss, eval_acc = model.evaluate(test_data)
print('\nEval Loss:{:.3f},Eval accuracy:{:.3f}'.format(eval_loss,eval_acc))


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_val_acc'], loc='best')
plt.savefig('/home/yi-hsien/ntnu/test1_acc.png')
plt.show()
plt.cla()
# summarize history for loss 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_val_loss'], loc='best') 
plt.savefig('/home/yi-hsien/ntnu/test1_loss.png')
plt.show()
plt.close()
