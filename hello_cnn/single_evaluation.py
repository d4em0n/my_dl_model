from keras.models import Sequential, model_from_json
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input, Activation, Dropout
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import pickle
import string

data = pd.read_csv("evaluation.csv", sep=",",  dtype={"nama":str,"country":int})
tokenizer = pickle.load(open("tokenizer.pc", "rb"))
X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(data['nama']), 42, padding='post')
#X_test = np.expand_dims(X_test, axis=2)
#Y_test = to_categorical(data['country'].values)
Y_test = data['country'].values

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
