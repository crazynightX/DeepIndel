import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Flatten,Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1

def Deepindel(bert_model):
   inputs = Input(shape=(179,),dtype=tf.int32)
   bert_score = bert_model(inputs)[0]
   dense = Dense(60,activation='relu',kernel_regularizer=l1(0.0001))(bert_score)
   flaten = Flatten()(dense)
   x = Dense(192,activation='relu')(flaten)
   x = Dropout(rate=0.4)(x)
   output = Dense(6,activation='sigmoid')(x)
   model = Model(inputs=[inputs],outputs=[output])
   return model

