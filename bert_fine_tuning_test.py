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
def encode_words(s):
  tokens = tokenizer.tokenize(s)
  tokens.append('[SEP]')
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  return token_ids[:511]

def bert_encode(data_to_be_encoded):
  content_list = tf.ragged.constant([encode_words(contents) for contents in data_to_be_encoded['content'].values])
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



##declare path
bert_model_path = '/home/yi-hsien/ntnu/fine_tuned_bert/bert_2'

#import/set_up tokenizer and model
tokenizer = BertTokenizer.from_pretrained("/home/yi-hsien/ntnu/bert_model_chinese_wwm_ext/publish")
loaded_model = tf.saved_model.load(bert_model_path)
probability_model = tf.keras.Sequential([tf.keras.layers.Softmax()])



#load entire news data, and process input dict
total_data = load_newsdata('/home/yi-hsien/ntnu/test_csv/apple_realtime200V1.csv')
glue_test = bert_encode(total_data)

predictions = loaded_model(glue_test)

print(predictions)









