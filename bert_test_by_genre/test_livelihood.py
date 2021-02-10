#this file is used to find how accurate a fine_tuned_bert model is to a specific news provider
#remember to change this accordingly

all_csv_names = ['applerealtime1250V1','cna2500V1','globalmilitary1000V1',
        'liberty1250V1','nooho400V1','udndaily1250V1','chinatimes1250V1',
        'ettoday2500V1','kknews1600V1','mission1000V1','qiqi1000V1']
all_csv_labels = [0,0,1,0,1,0,0,1,1,1]
all_csv_size = [1250,2500,1000,1250,400,1250,1250,2500,1600,1000,1000]
result_credibilities = []
#other things to change
'''
model_path
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
def encode_words(s):
  tokens = tokenizer.tokenize(s)
  tokens.append('[SEP]')
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  return token_ids[:127]


def one_time_content_encode(content_string):
  content_list = tf.ragged.constant([encode_words(content_string)])
  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*content_list.shape[0]
  content_list_ids = tf.concat([cls, content_list], axis=-1)

  input_mask = tf.ones_like(content_list_ids).to_tensor(shape=(None,128))
  input_type_ids = tf.zeros_like(content_list_ids).to_tensor(shape=(None,128))
  inputs = {
    'input_ids': content_list_ids.to_tensor(shape=(None,128)),
    'input_mask': input_mask,
    'input_type_ids': input_type_ids
  }
  return inputs



##declare path
bert_model_path = '/home/yi-hsien/ntnu/fine_tuned_bert/livelihood_1'

#import/set_up tokenizer and model
tokenizer = BertTokenizer.from_pretrained("/home/yi-hsien/ntnu/bert_model_chinese_wwm_ext/publish")
loaded_model = tf.saved_model.load(bert_model_path)
probability_model = tf.keras.Sequential([tf.keras.layers.Softmax()])




import numpy as np
import csv


for names in range(len(all_csv_names)):
    print("processing: {}".format(all_csv_names[names]))
    CSV_PATH = '/home/yi-hsien/ntnu/test_csv/livelihood/testingSet/'+all_csv_names[names]+'.csv'
    RESULT_PATH = '/home/yi-hsien/ntnu/test_csv_results/'+all_csv_names[names]+'.csv'
    
    #load entire news data, and process input dict
    total_data = load_newsdata(CSV_PATH)
    print(total_data)
    
    predictions = []
    #prediction process
    
    for samples in total_data['content']:
        processed_input = one_time_content_encode(samples)
        temp = probability_model.predict(loaded_model(processed_input)[0])
        predictions.append(temp)

        with open(RESULT_PATH, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(predictions)





    #check credibility
    accurate_count = 0
    total_count = 0
    for i in range(all_csv_size[names]):
        if np.argmax(predictions[i][0]) == all_csv_labels[names]:
            accurate_count+=1
        total_count += 1

    print("total:{}".format(total_count))
    print("credibility rate is {}%".format(accurate_count/all_csv_size[names]*100))
    result_credibilities.append(accurate_count/all_csv_size[names]*100)

print("total result-------------")
for names in range(len(all_csv_names)):
    print(all_csv_names[names]+": {:.2f}".format(result_credibilities[names]))





