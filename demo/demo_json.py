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

print("<br> -- 內文如下 -- <br>")
print("標題: ")
print(title)
print("<br>")
print("內文: ")
print(content)

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
print("此篇為假新聞之機率有: {:.3f}%<br>".format(predictions[0][1]*100))
if result == 1:
  print("認定為假新聞")
else:
  print("認定為真新聞")





