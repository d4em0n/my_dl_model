from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Activation, Dropout
from keras.utils import to_categorical
import numpy as np

train_dataset = np.loadtxt("train.csv", delimiter=",", converters={0:int})
X = train_dataset[:,0:24]
X = np.expand_dims(X, axis=2)
Y = to_categorical(train_dataset[:,24])

filters = 128
kernel_size = 3
model = Sequential()

model.add(Conv1D(filters, kernel_size, activation='relu', input_shape=(24, 1), strides=1))
model.add(Conv1D(96, kernel_size, activation='relu', input_shape=(24, 1), strides=1))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(20))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=100, batch_size=32)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

test_dataset = np.loadtxt("evaluation.csv", delimiter=",", converters={0:int})
X_test = test_dataset[:,0:24]
X_test = np.expand_dims(X_test, axis=2)
Y_test = to_categorical(test_dataset[:,24])
score = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
