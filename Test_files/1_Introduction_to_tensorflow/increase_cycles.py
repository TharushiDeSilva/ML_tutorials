from statistics import mode
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(units = 1, input_shape = [1])])
model.compile(optimizer='sgd', loss='mean_squared_error') 

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=700)
print(model.predict([10.0]))
print(model)

# original answer = 18.977282 (error - 0.022718)

# increase the run times from 500 to 700
#answer 18.997412  (error - 0.002588)

 