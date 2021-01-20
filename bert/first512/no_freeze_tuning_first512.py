#take 512 at a time, read first part of the text only
######################################
#remember to change the model location
######################################
import tensorflow as tf
import tensorflow_datasets
from transformers import *
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("/home/yi-hsien/ntnu/bert_model_chinese_wwm_ext/publish")

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

  input_mask = tf.ones_like(content_list_ids).to_tensor()
  input_type_ids = tf.zeros_like(content_list_ids).to_tensor()
  inputs = {
    'input_ids': content_list_ids.to_tensor(),
    'input_mask': input_mask,
    'input_type_ids': input_type_ids
  }
  return inputs




total_data = load_newsdata('/home/csliao/tf01/dataset/adclu2nmqgk82R.csv').sample(frac=1).reset_index(drop=True)

train_data = total_data.sample(frac = 0.9)
validation_data = total_data.drop(train_data.index)

glue_train = bert_encode(train_data)
glue_train_labels = tf.convert_to_tensor(train_data['labeled'])


glue_validation = bert_encode(validation_data)
glue_validation_labels = validation_data['labeled']

for key, value in glue_train.items():
  print(f'{key:15s} shape: {value.shape}')

print(f'glue_train_labels shape: {glue_train_labels.shape}')


#########data processing finished ########

bert_config = BertConfig.from_json_file('/home/yi-hsien/ntnu/bert_model_chinese_wwm_ext/publish/bert_config.json')
model = TFBertForSequenceClassification(bert_config)

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(
    '/home/yi-hsien/ntnu/bert_model_chinese_wwm_ext/publish/bert_model.ckpt').assert_consumed()


# Set up epochs and steps
epochs = 4
batch_size = 5
eval_batch_size = 32

train_data_size = len(glue_train_labels)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

# creates an optimizer with learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)


#model.summary()

model.fit(glue_train,glue_train_labels,validation_data=(glue_validation,glue_validation_labels),batch_size=batch_size,epochs=epochs,
          steps_per_epoch=steps_per_epoch)


#tf.saved_model.save(model,'/home/yi-hsien/ntnu/fine_tuned_bert/bert_23')
tf.saved_model.save(model,'/home/csliao/tf01/yi-hsien/bert_23')







import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')
# summarize history for accuracy
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_val_acc'], loc='best')
#plt.savefig('/home/yi-hsien/ntnu/fine_tune_acc.png')
plt.savefig('/home/csliao/tf01/yi-hsien/fine_tune_acc.png')

plt.show()
plt.cla()
# summarize history for loss 
plt.plot(model.history.history['loss']) 
plt.plot(model.history.history['val_loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_val_loss'], loc='best') 
#plt.savefig('/home/yi-hsien/ntnu/fine_tune_loss.png')
plt.savefig('/home/csliao/tf01/yi-hsien/fine_tune_loss.png')
plt.show()
plt.close()


