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

#Eric Hsieh 2020.09.28
#using huggingface_transformers https://github.com/huggingface/transformers
#using chinese_bert_wwm https://github.com/ymcui/Chinese-BERT-wwm

import tensorflow as tf
#print("tensorflow imported")
#print("<br>")
import tensorflow_datasets
#print("tensorflow_datsets imported")
#print("<br>")
from transformers import *
#print("transformers imported")
#print("<br>")

#tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
tokenizer = BertTokenizer.from_pretrained("/home/yi-hsien/ntnu/bert_model_chinese_wwm_ext/publish")

#print("bertmodel initialized")
#print("<br>")

df = [title, content]

###encoding###
def encode_words(s):
  tokens = tokenizer.tokenize(s)
  #tokens.append('[SEP]')
  return tokenizer.convert_tokens_to_ids(tokens)
input_title = encode_words(df[0])
input_content = encode_words(df[1])

#print("done encoding")
#print("<br>")

###choose testing range: title or content or both###
total_content = [ input_content ]
#print("content chose")
#print("<br>")

###load model###
test_model = tf.keras.models.load_model('/home/yi-hsien/ntnu/NTNU_GIMC_FakeNewsDetector/models/200928-1_model.h5')
#print("model loaded")
#print("<br>")

###make prediction###
probability_model = tf.keras.Sequential([test_model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(total_content)
#print("prediction made")
#print("<br>")

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

def pad(X):
  pad_num = text_len
  if len(X[0])> pad_num :
    return [X[0][:pad_num]]
  else:
    for i in range(len(X)):
      X[i] = X[i] + [0]*(pad_num - len(X[i]))
    return X



def predict_one(X): #take in a [""]
  total_content = pad([encode_words(X)])
  probability_model = tf.keras.Sequential([test_model,tf.keras.layers.Softmax()])
  lime_predictions = probability_model.predict(total_content)
  #print(lime_predictions)
  return lime_predictions[0][1] #return the percentage of fakeness

def predict_proba(X): #should take a list instead of one
  content = X
  #print(content) 
  return np.array([[float(1 - predict_one(x)), float(predict_one(x))] for x in content])

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
  for items in exp_list:
    print(items)
    print("<br>")
else:
  print("無明顯特徵，重新整理以獲取更多結果... <br>")




