import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

plt.style.use('seaborn-white')

songs = pd.read_csv('training_data.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()


#Randomly picking a number of rows dividing them into training and test set.
trainI = np.random.choice(songs.shape[0], size = 350, replace = False)
trainIndex = songs.index.isin(trainI)
train = songs.iloc[trainIndex]      #Training set
test = songs.iloc[~trainIndex]      #test set


model = BaggingClassifier()

X_train = train[['acousticness', 'danceability', 'duration', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']]
Y_train = train['label']

X_test = test[['acousticness', 'danceability', 'duration', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']]
Y_test = test['label']

model.fit(X_train, Y_train)
print('model summary:')
print(model)

predict_prob = model.predict_proba(X_test)
print('The class order in the model:')
print(model.classes_)
predict_prob[0:5] # inspect  the first 5 predictions

prediction = np.empty(len(X_test), dtype=object)
prediction = np.where(predict_prob[:,0]>=0.5, 'like', 'unlike')
print(prediction[0:5])  # Inspect the first 5 predicitons after labeling.

print(pd.crosstab(prediction, Y_test))

Y_new = np.empty(len(X_test), dtype=int)
Y_new = np.where(Y_test[:]==0, 'like', 'unlike')

print(np.mean(prediction == Y_new))
#print(prediction)
#print(Y_new)
