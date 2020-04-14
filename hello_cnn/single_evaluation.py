from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import model_from_json
import numpy as np
import pickle
import string
import sys

MAX_NAME = 24 # max name length

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

countries = ["russian", "chinese", "arabic", "germany"]
tokenizer = pickle.load(open("tokenizer.pc", "rb"))

while True:
    print("Enter the name you want to classify: ", flush=True, end="")
    name = input("")
    X = sequence.pad_sequences(tokenizer.texts_to_sequences([name]), 24).reshape(24,1)
    res = loaded_model.predict(np.array([X,]))[0]
    for i in range(len(res)):
        print("probability %s names = %.2f%%" % (countries[i], float(res[i]*100)))
