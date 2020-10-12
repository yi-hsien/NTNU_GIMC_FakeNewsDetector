#Eric Hsieh 2020.10.04
#using huggingface_transformers https://github.com/huggingface/transformers
#using chinese_bert_wwm https://github.com/ymcui/Chinese-BERT-wwm

#this file is used to find how accurate a model is to a specific news provider
#remember to change this accordingly
label_due_to_news_provider = 1
range_due_to_dataset = 200

#other things to change
'''
model_path
data_name
data_result_name
tokenizer_path
'''


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


CSV_PATH = "/home/yi-hsien/ntnu/test_csv/mission200V1.csv"

import pandas as pd
def load_newsdata():
    # load raw source data
    RAW_CSV = CSV_PATH
    df = pd.read_csv(RAW_CSV, sep=',', encoding='utf8')
    return(df)

df = load_newsdata()
df1 = df[['title', 'content', 'labeled']]




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
input_title_list = [[encode_words(titles)] for titles in df1['title'].values]
input_content_list = [[encode_words(contents)] for contents in df1['content'].values]



print("done encoding")
print("<br>")


total_content = [ input_content_list ]
print("content chose")
print("<br>")



test_model = tf.keras.models.load_model('/home/yi-hsien/ntnu/NTNU_GIMC_FakeNewsDetector/models/201011_model.h5')
print("model loaded")
print("<br>")





probability_model = tf.keras.Sequential([test_model,tf.keras.layers.Softmax()])
predictions = []
for samples in total_content[0]:
  temp = probability_model.predict(samples)
  predictions.append(temp)

print("prediction made")
print("<br>")
#print(predictions)


import numpy as np
import csv
with open('/home/yi-hsien/ntnu/test_csv_results/mission200V1_results.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(predictions)

#check credibility
accurate_count = 0
for i in range(range_due_to_dataset):
  if np.argmax(predictions[i][0]) == label_due_to_news_provider:
    accurate_count+=1

print("credibility rate is {}%".format(accurate_count/range_due_to_dataset*100))