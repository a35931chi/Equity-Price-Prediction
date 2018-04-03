import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Input sequence
wholeSequence = [[0,0,0,0,0,0,0,0,0,2,1],
                 [0,0,0,0,0,0,0,0,2,1,0],
                 [0,0,0,0,0,0,0,2,1,0,0],
                 [0,0,0,0,0,0,2,1,0,0,0],
                 [0,0,0,0,0,2,1,0,0,0,0],
                 [0,0,0,0,2,1,0,0,0,0,0],
                 [0,0,0,2,1,0,0,0,0,0,0],
                 [0,0,2,1,0,0,0,0,0,0,0],
                 [0,2,1,0,0,0,0,0,0,0,0],
                 [2,1,0,0,0,0,0,0,0,0,0]]

# Preprocess Data:
wholeSequence = np.array(wholeSequence, dtype=float) # Convert to NP array.
data = wholeSequence[:-1] # all but last

target = wholeSequence[1:] # all but first


# Reshape training data for Keras LSTM model
# The training data needs to be (batchIndex, timeStepIndex, dimentionIndex)
# Single batch, 9 time steps, 11 dimentions
data = data.reshape((1, 9, 11))
target = target.reshape((1, 9, 11))
print(data.shape)
print(data)
print(target.shape)
print(target)
what = input('what')
# Build Model
model = Sequential()  
model.add(LSTM(11, input_shape=(9, 11), unroll=True, return_sequences=True))
model.add(Dense(11))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(data, target, nb_epoch=2000, batch_size=1, verbose=2)
