from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import numpy

# fix random seed for reproducibility
numpy.random.seed(1234)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
# print(X)
Y = dataset[:,8]
# print(Y)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# make a prediction
test_dataset = dataset[5,0:8]
y_pred = model.predict_classes(test_dataset.reshape(1, 8))
print(y_pred)

# confusion matrix
print("Confusion Matrix")
y_dataset_pred = model.predict_classes(X)
print(confusion_matrix(Y, y_dataset_pred))
