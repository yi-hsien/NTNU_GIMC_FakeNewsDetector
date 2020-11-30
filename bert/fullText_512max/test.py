#this file is used to find how accurate a fine_tuned_bert model is to a specific news provider
#remember to change this accordingly
label_due_to_news_provider = 0
range_due_to_dataset = 200

#other things to change
'''
model_path
data_name
data_result_name
tokenizer_path
'''
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

def encode_words(s,content_list):
  if s is None:
    return
  tokens = tokenizer.tokenize(s)
  tokens.append('[SEP]')
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  content_list.append(token_ids[:511])
  while len(token_ids)>511:
    token_ids = token_ids[512:]
    content_list.append(token_ids[:511])

def one_time_bert_encode(string_to_be_encoded):
  content_list = []
  encode_words(string_to_be_encoded,content_list)
  content_list = tf.ragged.constant(content_list)
  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*content_list.shape[0]
  content_list_ids = tf.concat([cls, content_list], axis=-1)
  input_mask = tf.ones_like(content_list_ids).to_tensor(shape=(None,512))
  input_type_ids = tf.zeros_like(content_list_ids).to_tensor(shape=(None,512))
  inputs = {
    'input_ids': content_list_ids.to_tensor(shape=(None,512)),
    'input_mask': input_mask,
    'input_type_ids': input_type_ids
  }
  return inputs

##declare path
bert_model_path = '/home/yi-hsien/ntnu/fine_tuned_bert/bert_5'

#import/set_up tokenizer and model
tokenizer = BertTokenizer.from_pretrained("/home/yi-hsien/ntnu/bert_model_chinese_wwm_ext/publish")
loaded_model = tf.saved_model.load(bert_model_path)
probability_model = tf.keras.Sequential([tf.keras.layers.Softmax()])



#load entire news data, and process input dict
total_data = load_newsdata('/home/yi-hsien/ntnu/test_csv/chinatimes200V1.csv')

print(total_data)



predictions = []
#prediction process
for samples in total_data['content']:
    processed_input = one_time_bert_encode(samples)
    raw_prediction_list = loaded_model(processed_input)
    string_prediction = []
    for i in range(len(raw_prediction_list)):
        temp = probability_model.predict(raw_prediction_list[i])
        string_prediction.append(temp)
    predictions.append(string_prediction)
    
print("\n\n\n\npredictions:")
print(predictions)
print("\n\n\n\n")


import numpy as np
import csv
with open('/home/yi-hsien/ntnu/test_csv_results/chinatimes200V1.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     for i in range(len(predictions)):
        wr.writerow(predictions[i])


#check credibility
accurate_count = 0
for i in range(range_due_to_dataset):
    final_decision = 0
    for j in range(len(predictions[i])):
        if np.argmax(predictions[i][j]) == 1 : 
            final_decision = 1
    if final_decision == label_due_to_news_provider:
        accurate_count+=1

print("credibility rate is {}%".format(accurate_count/range_due_to_dataset*100))
