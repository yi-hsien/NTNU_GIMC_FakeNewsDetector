#import tensorflow as tf
#import tensorflow_datasets
#from transformers import *

#tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
#model = BertModel.from_pretrained("hfl/chinese-bert-wwm")

'''
test_string = ['駐']
test_u = r'"\udce9\udca7\udc90"'


print(test_string)
#print(test_u)

import json

test_u = json.loads(test_u)
print(test_u)
'''

test = '"\udce9\udca7\udc90"'


import json

print(json.loads(test))