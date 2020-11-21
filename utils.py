import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.metrics import confusion_matrix,classification_report
import os


max_words = 2000
embed_dim = 128
lstm_out = 196
batch_size=32


class ReadData:
	def __init__(self):
		pass
	def read_airline(self):
		data=pd.read_csv('airline_sentiment_analysis.csv')
		data=data[['airline_sentiment','text']]
		return data


class Encoding_sentiment:
	def __init__(self,data):
		self.data=data
	
	def encoding(self):
		return pd.get_dummies(self.data['airline_sentiment'],drop_first=True).values


class Preprocessing:
	def __init__(self):
		pass
		
	def get_lower_regex(self,text):
		self.text= self.text.apply(lambda x: x.lower())
		self.text = self.text.apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
		return self.text
	
	def remove_company_tag(self,text):
		return pd.Series([x.split(maxsplit=1)[1] for x in self.text])
	
	def tokenization_padding(self,text):
		tokenizer = Tokenizer(num_words=max_words, split=' ')
		tokenizer.fit_on_texts(self.text.values)
		with open('models/tokenizer.pickle', 'wb') as handle:
			pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
		self.text=tokenizer.texts_to_sequences(self.text.values)
		self.text=pad_sequences(self.text)
		return self.text
		
class Train_Process(Preprocessing):
	def __init__(self):
		pass
	
	def dataset_train_preprocess(self,data):
		self.text=data['text']
		self.text= self.get_lower_regex(self.text)
		self.text= self.remove_company_tag(self.text)
		self.text= self.tokenization_padding(self.text)
		return self.text
	
	def single_test_preprocess(self,text):
		self.text=text
		self.text= self.get_lower_regex(self.text)
		self.text= self.tokenization_padding(self.text)
		return self.text
		

class architecture:
	
	def __init__(self):
		pass
	
	
	def lstm_model(self):
		model = Sequential()
		model.add(Embedding(max_words, embed_dim,input_length = 31))
		model.add(SpatialDropout1D(0.4))
		model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
		model.add(Dense(1,activation='sigmoid'))
		model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
		return model


class Train_Data_Split:
	
	def __init__(self,text,outcome):
		self.features=text
		self.outcome=outcome
	
	def train_split(self):
		X_train, x_test, Y_train, y_test = train_test_split(self.features,self.outcome, test_size = 0.34, random_state = 42)
		return X_train, x_test, Y_train, y_test
	
class Valid_Data_Split(Train_Data_Split):
	
	def valid_test_split(self):
		X_test, X_valid, Y_test, Y_valid = train_test_split(self.features,self.outcome, test_size = 0.5, random_state = 42)
		return X_test, X_valid, Y_test, Y_valid
	
		

class Train_model:

	def __init__(self):
		pass
	
	def fit(self,model,X_train,Y_train,X_valid,Y_valid):
		checkpt= ModelCheckpoint('models/model.h5',monitor='val_accuracy', save_best_only=True, mode='max',verbose=0)
		history= model.fit(X_train, Y_train,validation_data=(X_valid,Y_valid), epochs = 1, batch_size=batch_size, verbose = 0,callbacks=[checkpt])
		return history  



class Testing:
	def __init__(self):
		pass
	def testing_metrics(self,model,X_test,Y_test):
		predict=model.predict_classes(X_test,batch_size=32,verbose=0)
		print(classification_report(Y_test,predict))
		print(confusion_matrix(Y_test,predict))       