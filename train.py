import os
from utils import *

os.makedirs('models')

# Reading the data
data=ReadData().read_airline()

# Encoding the Categorical Sentiments
Y=Encoding_sentiment(data).encoding()

# Preprocessing the airline reviews
Preprocess_object=Train_Process()

X=Preprocess_object.dataset_train_preprocess(data)

# Instantiate the LSTM Model
model=architecture().lstm_model()

# Training and Validation Splitting
X_train, x_test_valid, Y_train, y_test_valid= Train_Data_Split(X,Y).train_split()
X_test, X_valid, Y_test, Y_valid = Valid_Data_Split(x_test_valid,y_test_valid).valid_test_split()

# Training the model
training=Train_model()
history=training.fit(model,X_train,Y_train,X_valid,Y_valid)

# Testing the model
Test=Testing().testing_metrics(model,X_test,Y_test)
