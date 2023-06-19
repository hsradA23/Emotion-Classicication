from transformers import AutoTokenizer,TFBertModel
import shutil
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertConfig,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
max_len=70
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

bert = TFBertModel.from_pretrained('bert-base-cased')
tokenizer.save_pretrained('bert-tokenizer')
bert.save_pretrained('bert-model')
shutil.make_archive('bert-model','zip','bert-model')
shutil.make_archive('bert-tokenizer', 'zip', 'bert-tokenizer')
dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')



input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

embeddings = bert(input_ids,attention_mask = input_mask)[0] 
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = tf.keras.layers.Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = tf.keras.layers.Dense(32,activation = 'relu')(out)

y = tf.keras.layers.Dense(6,activation = 'sigmoid')(out)
    
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True


def the_machine(texts):
  model.load_weights(f'/home/adarsh/ai_project/sentiment_weights.h5')
  x_val = tokenizer(
    text=texts,
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True) 
  validation = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
  validation
  encoded_dict  = {'anger':0,'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}
  b=dict()
  for key , value in zip(encoded_dict.keys(),validation[0]):
    b[key]=value
  return max(b, key=b.get)
