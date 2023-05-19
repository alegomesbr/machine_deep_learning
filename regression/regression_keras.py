import numpy as np
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(1337)  # for reproducibility

dataset = np.loadtxt("housing.csv", delimiter=",")

# split into input (X) and output (Y) variables
X_train = dataset[:,0:13]
# print(X_train)
Y_train = dataset[:,13]
# print(Y_train)

model = Sequential()
model.add(Dense(4, input_dim=13, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))

#mae: Mean Absolute Erro
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

print('Training -----------')
model.fit(X_train, Y_train, epochs=1000, verbose=0)

# evaluate the model
scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#W, b = model.layers[0].get_weights()
#print('Weights=', W, '\nbiases=', b)

print('Testing -----------')
test_dataset = dataset[8,0:13]
y_pred = model.predict(test_dataset.reshape(1, 13))
print(y_pred)
