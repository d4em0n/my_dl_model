from keras.models import Sequential, model_from_json
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input, Activation, Dropout
import numpy as np
import pandas as pd

test_dataset = np.loadtxt("test_name_ru_cn.csv", delimiter=",", converters={0:int})
X_test = test_dataset[:,0:24]
X_test = np.expand_dims(X_test, axis=2)
Y_test = test_dataset[:,24]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
