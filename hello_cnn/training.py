from keras.models import Sequential
from keras.layers import Conv1D, Dense, GlobalMaxPooling1D, Flatten, Activation, Dropout, MaxPool1D, Embedding
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
import pickle
import string

input_length = 42

data = pd.read_csv("train.csv", sep=",",  dtype={"nama":str,"country":int})
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data['nama'])
pickle.dump(tokenizer, open("tokenizer.pc", "wb+")) # save tokenizer object
X = sequence.pad_sequences(tokenizer.texts_to_sequences(data['nama']), input_length, padding='post')
#X = np.expand_dims(X, axis=2)
Y = data['country'].values
#Y = to_categorical(data['country'].values)
filters = 148
kernel_size = 4
model = Sequential()

model.add(Embedding(len(tokenizer.word_index) + 1, 48, input_length=input_length))
model.add(Conv1D(128, 5, activation='relu', strides=1))
model.add(Conv1D(256, 4, activation='relu', strides=1))
model.add(Flatten())
model.add(Dense(96))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(96))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(13, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=110, batch_size=120)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

data = pd.read_csv("evaluation.csv", sep=",",  dtype={"nama":str,"country":int})
X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(data['nama']), input_length, padding='post')
#X_test = np.expand_dims(X_test, axis=2)
Y_test = data['country'].values
#Y_test = to_categorical(data['country'].values)
score = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
