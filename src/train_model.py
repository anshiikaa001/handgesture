import pandas as pd
from sklearn.neural_network import MLPClassifier
import pickle

data = pd.read_csv('model/keypoint.csv', header=None)

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

model = MLPClassifier(hidden_layer_sizes=(150,100), max_iter=1000)
model.fit(X, y)

pickle.dump(model, open('model/model.pkl', 'wb'))

print("Model trained successfully ✅")