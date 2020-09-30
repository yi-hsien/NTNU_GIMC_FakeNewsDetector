import sys
demo = sys.argv[0]
title = sys.argv[1]
content = sys.argv[2]
print("<br>")
print(title)
print("<br>")
print(content)
print("<br>")

print("ok, registered")
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
#tokenizer = BertTokenizer.from_pretrained("/home/yi-hsien/ntnu/bert_model_chinese_wwm_ext/publish")

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

'''
input_title = encode_words(df[0])
input_content = encode_words(df[1])

print("done encoding")
print("<br>")
'''



test_model = tf.keras.models.load_model('/home/yi-hsien/ntnu/NTNU_GIMC_FakeNewsDetector/models/200928-1_model.h5')
#test_model = tf.keras.models.load_model('/home/yi-hsien/ntnu/NTNU_GIMC_FakeNewsDetector/test.h5')
print("model loaded")
print("<br>")


probability_model = tf.keras.Sequential([test_model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(content)
print("prediction made")
print("<br>")
print(predictions)



import numpy as np

#print("real news_percentage={}, fake news_percentage={}".format(predictions[0],predictions[1]))
print(np.argmax(predictions))




