import sys
demo = sys.argv[0]
title = sys.argv[1]
content = sys.argv[2]
print("<br>")
print(title)
print("<br>")
print(content)
print("<br>")

import json
'''
the task here..
things sent in will look like \\udc....

which is a json encoding that needs to be decoded with json.loads()
or other json decoding methods

but for now we gotta use raw string literal to change the string into raw string first
which the string object doesn't seem to do this

1. keep on trying methods on the string object 
2. creating a new file to put the input string, and import it thru file path


'''


print("ok, registered")
print("<br>")

for i in range(3):
  print("<br>")

print("string length: ")
print(len(content))
print("<br>")
print("string type: ")
print(type(content))



for i in range(3):
  print("<br>")


#Eric Hsieh 2020.09.28
#using huggingface_transformers https://github.com/huggingface/transformers
#using chinese_bert_wwm https://github.com/ymcui/Chinese-BERT-wwm

import tensorflow as tf
print("tensorflow imported")
print("<br>")
import tensorflow_datasets
print("tensorflow_datsets imported")
print("<br>")
from transformers import *
print("transformers imported")
print("<br>")

#tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
tokenizer = BertTokenizer.from_pretrained("/home/yi-hsien/ntnu/bert_model_chinese_wwm_ext/publish")

print("bertmodel initialized")
print("<br>")

df = [title, content]

#encoding
def encode_words(s):
  tokens = tokenizer.tokenize(s)
  #tokens.append('[SEP]')
  return tokenizer.convert_tokens_to_ids(tokens)
'''
title = tf.ragged.constant(encode_words(df[0]))
content = tf.ragged.constant(encode_words(df[1]))

cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]
content = tf.concat([cls,content], axis=-1)
content_tensor = contents_list.to_tensor()
'''

input_title = encode_words(df[0])
input_content = encode_words(df[1])


print("done encoding")
print("<br>")

total_content = [ input_content ]
print("content chose")
print("<br>")






test_model = tf.keras.models.load_model('/home/yi-hsien/ntnu/NTNU_GIMC_FakeNewsDetector/models/200928-1_model.h5')
#test_model = tf.keras.models.load_model('/home/yi-hsien/ntnu/NTNU_GIMC_FakeNewsDetector/test.h5')
print("model loaded")
print("<br>")


probability_model = tf.keras.Sequential([test_model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(total_content)
print("prediction made")
print("<br>")
print(predictions)



import numpy as np

#print("real news_percentage={}, fake news_percentage={}".format(predictions[0],predictions[1]))
print(np.argmax(predictions))




