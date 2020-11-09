import sys
#argv import
demo = sys.argv[0]
title = sys.argv[1]
content = sys.argv[2]
input_array = sys.argv[1:]
#print("<br> parameters imported")

###encoding transfer###

##title
#title = '\''+title+'\''
#print("<br>original length: ")
#print(len(title))
#print("<br>")
title = title.encode('utf-8', errors='surrogateescape').decode('utf-8')
#print(title)
#print("<br>length: ")
#print(len(title))

##content
#content = '\''+content+'\''
#print("<br>original length: ")
#print(len(content))
#print("<br>")
content = content.encode('utf-8', errors='surrogateescape').decode('utf-8')
#print(content)
#print("<br>length: ")
#print(len(content))
#print("<br>done decoding<br><br>")
print("<br><span style='font-size:20px'> -- 內文如下 -- <br>")
print("標題: ")
print(title)
print("<br>")
print("內文: ")
print(content)
print("</span>")
print("<span style='font-size:27px'>")
for i in range(4):
  print("<br>")

#Eric Hsieh 2020.11.08
#using huggingface_transformers https://github.com/huggingface/transformers
#using chinese_bert_wwm https://github.com/ymcui/Chinese-BERT-wwm

##declare path
bert_model_path = '/home/yi-hsien/ntnu/fine_tuned_bert/bert_1'
##

import tensorflow as tf
import tensorflow_datasets
from transformers import *
import pandas as pd


def load_newsdata(RAW_CSV):
    # load raw source data
    pd.read_csv(RAW_CSV,sep=',',encoding='utf8')
    df = pd.read_csv(RAW_CSV, sep=',', encoding='utf8')
    df = df.sample(frac=1).reset_index(drop=True)
    return(df[['content', 'labeled']]) #returns content and label
def encode_words(s):
  tokens = tokenizer.tokenize(s)
  tokens.append('[SEP]')
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  return token_ids[:127]

def bert_encode(data_to_be_encoded):
  content_list = tf.ragged.constant([encode_words(contents) for contents in data_to_be_encoded['content'].values])
  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*content_list.shape[0]
  content_list_ids = tf.concat([cls, content_list], axis=-1)

  input_mask = tf.ones_like(content_list_ids).to_tensor()
  input_type_ids = tf.zeros_like(content_list_ids).to_tensor()
  inputs = {
    'input_ids': content_list_ids.to_tensor(),
    'input_mask': input_mask,
    'input_type_ids': input_type_ids
  }
  return inputs

def one_time_content_encode(content_string):
  content_list = tf.ragged.constant([encode_words(content_string)])
  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*content_list.shape[0]
  content_list_ids = tf.concat([cls, content_list], axis=-1)

  input_mask = tf.ones_like(content_list_ids).to_tensor()
  input_type_ids = tf.zeros_like(content_list_ids).to_tensor()
  inputs = {
    'input_ids': content_list_ids.to_tensor(),
    'input_mask': input_mask,
    'input_type_ids': input_type_ids
  }
  return inputs

#import/set_up tokenizer and model
tokenizer = BertTokenizer.from_pretrained("/home/yi-hsien/ntnu/bert_model_chinese_wwm_ext/publish")
loaded_model = tf.saved_model.load(bert_model_path)
probability_model = tf.keras.Sequential([tf.keras.layers.Softmax()])

#process data
processed_input = one_time_content_encode(content)

#make prediction
predictions = probability_model.predict(loaded_model(processed_input)[0])

import numpy as np
print("real news_percentage={:.3f}%, fake news_percentage={:.3f}%<br>".format(predictions[0][0]*100,predictions[0][1]*100))
result = np.argmax(predictions)
print("此篇為假新聞之機率有: {:.3f}%<br></span>".format(predictions[0][1]*100))
if result == 1:
  print("<span style='font-size:40px'>認定為假新聞</span>")
else:
  print("<span style='font-size:40px'>認定為真新聞</span>")




###################################
##############lime#################
###################################
from lime import lime_text
from lime.lime_text import LimeTextExplainer
feature_found = False


text = content
text_len = len(content)

def predict_one(X): #take in a [""]
  one_time_content = X[0]
  processed_input = one_time_content_encode(one_time_content)
  lime_predictions = probability_model.predict(loaded_model(processed_input)[0])
  #print(lime_predictions)
  return lime_predictions[0][1] #return the percentage of fakeness


def predict_proba(X): #should take a list instead of one
  content = X
  #print(content) 
  one_time_prediction = predict_one(X)
  return np.array([[float(1 - one_time_prediction), float(one_time_prediction)] for x in content])

explainer = LimeTextExplainer(class_names=["real", "fake"])
exp = explainer.explain_instance(
    text, predict_proba, num_features=6, num_samples = 10
)


exp_list = exp.as_list()
result = exp.as_html()
trimmed_result = result[6:-7]
print(trimmed_result)

if abs(exp_list[0][1]) > 0.02:
  print("明顯特徵為:<br>")
  for (item,weight) in exp_list:
    if abs(weight) > 0.02:
      print(item)
      print("比重約")
      print(weight)
      print("<br>")
else:
  print("無明顯特徵，重新整理以獲取更多結果... <br>")



