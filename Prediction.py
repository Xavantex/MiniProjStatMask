import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

plt.style.use('seaborn-white')

songs = pd.read_csv('training_data.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

tsongs = pd.read_csv('songs_to_classify.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()


model = BaggingClassifier()

X_train = songs[['acousticness', 'danceability', 'duration', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']]
Y_train = songs['label']

X_test = tsongs[['acousticness', 'danceability', 'duration', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']]


model.fit(X_train, Y_train)
print('model summary:')
print(model)

predict_prob = model.predict_proba(X_test)
print('The class order in the model:')
print(model.classes_)
print(predict_prob[0:5,0]) # inspect  the first 5 predictions

prediction = np.empty(len(X_test), dtype=object)
prediction = np.where(predict_prob[:,0]>=0.5, '0', '1')
print(prediction[0:5])  # Inspect the first 5 predicitons after labeling.

df = ''.join(prediction)

print(df)
