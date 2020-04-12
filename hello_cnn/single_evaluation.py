from keras.models import model_from_json
import numpy as np
import sys

MAX_NAME = 24 # max name length
name = "xi jinping"
if len(sys.argv) > 1:
    name = sys.argv[1][:MAX_NAME]

X = list(map(lambda i: float(ord(i)), name.lower().ljust(24, "\x00")))
X = np.array(X).reshape(24, 1)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

countries = ["russian", "chinese", "arabic"]
res = loaded_model.predict(np.array([X,]))[0]
for i in range(len(res)):
    print("probability %s names = %.2f%%" % (countries[i], float(res[i]*100)))
