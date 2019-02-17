import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

plt.style.use('seaborn-white')

songs = pd.read_csv('training_data.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

X = songs.drop(columns=['label'])
Y = songs['label']

#Randomly picking a number of rows dividing them into training and test set.

randomize_indices = np.random.choice(X.shape[0], X.shape[0], replace=False)

misclassification = np.zeros((10))

model = RandomForestClassifier()

for i in range(10):
    n = np.ceil(X.shape[0]/10) # number of samples in each fold
    validationIndex = np.arange(i*n, min(i*n+n, X.shape[0]), 1).astype('int')
    randomize_validationIndex = randomize_indices[validationIndex]
    X_train = X.iloc[~X.index.isin(randomize_validationIndex)]
    Y_train = Y.iloc[~Y.index.isin(randomize_validationIndex)]
    X_validation = X.iloc[randomize_validationIndex]
    Y_validation = Y.iloc[randomize_validationIndex]
    model.fit(X_train, Y_train)
    prediction = model.predict(X_validation)
    misclassification[i] = (np.mean(prediction != Y_validation))


mean = np.mean(misclassification)

print(mean)

plt.boxplot(misclassification)
plt.title('cross validation error for Random Forests')
plt.ylabel('validation error')
plt.show()
