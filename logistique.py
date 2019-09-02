import pandas as pd 
data = pd.read_csv('data_tp_reg_log.csv')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams.update({'figure.figsize': (10, 10)})
data.head()
plt.scatter(data['size'], data['is_malignant'], c=data['is_malignant'])
plt.title('Tumeur maligne en fonction de la taille')
plt.xlabel('Taille (micrometre)')
plt.ylabel('Statut tumeur')
plt.scatter(x='size', y='is_malignant', data=data)
from sklearn.linear_model import LogisticRegression
import numpy as np
model = LogisticRegression(random_state=0)
model.fit(data[['size']], data['is_malignant'])
model.score(data[['size']], data['is_malignant'])
to_predict = np.array([[0.414], [0.001], [1.1], [2000]])
# print(to_predict.shape)
model.predict(to_predict)

from sklearn.model_selection import train_test_split
X = data[['size']]
Y = data['is_malignant']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
training_set = [X_train], Y_train


problemes de dimensions X_train et Y_train ont des dimensions opposées 
model = LogisticRegression(solver='liblinear')
model.fit(X_train,Y_train)
model.predict(training_set)
model.score(X_train, Y_train)
model.score(X_test, Y_test)
import pickle
filename = 'finalized_model.pkl'
pickle.dump(model, open(filename, 'wb'))
print(model.predict([[1.8]])))