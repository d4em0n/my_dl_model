from keras.models import model_from_json
import numpy as np
import sys

MAX_NAME = 24 # max name length
name = "xi jinping"
if len(sys.argv) > 1:
    name = sys.argv[0][:MAX_NAME]
X = list(map(lambda i: float(ord(i)), name.lower().ljust(24, "\x00")))
X = np.array(X).reshape(24, 1)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

res = loaded_model.predict(np.array([X,]))
if res < 0.5:
    print("This is chinese name")
else:
    print("This is russian name")
