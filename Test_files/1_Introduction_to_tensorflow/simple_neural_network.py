#inputs(x) = [-1, 0, 1, 2, 3, 4]
#outputs(y) = [-3, -1, 1, 3, 5, 7] 
#determine the rule (Y = 2X -1)

from statistics import mode
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# define the simplest NN. Number of lines inside Sequential() defines the number of layers. here only one 
# Dense means a set of densely connected neurons , we have 1 dense layer(units), 
# input_shape is what type of input we have (x is one dimensioanl data)
model = Sequential([Dense(units = 1, input_shape = [1])])

# the model guesses the relationship (ex- Y = 10X +10) and it tcalculates the loss
#sgd = shocastic gradient descent ( makes a new guess by minimizing the loss)
model.compile(optimizer='sgd', loss='mean_squared_error') 

# arrange data and results into numpy arrays
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# The training begins with fit() command. try 500 times 
model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))
print(model)

# The answer will not exactly be 19. it's a close value because after 500 runs, our loss is not zero. 
# multiple runs does not give the same answer 

 