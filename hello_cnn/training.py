from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Activation, Dropout
import numpy as np

train_dataset = np.loadtxt("train_name_ru_cn.csv", delimiter=",", converters={0:int})
X = train_dataset[:,0:24]
X = np.expand_dims(X, axis=2)
Y = train_dataset[:,24]

filters = 100
kernel_size = 3
model = Sequential()

model.add(Conv1D(filters, kernel_size, activation='relu', input_shape=(24, 1), strides=1))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(18))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=100, batch_size=32)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
