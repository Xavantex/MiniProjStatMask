import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import sklearn.discriminant_analysis as skl_da

cross = 5

plt.style.use('seaborn-white')

songs = pd.read_csv('training_data.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

X = songs.drop(columns=['label'])
Y = songs['label']

randomize_indices = np.random.choice(X.shape[0], X.shape[0], replace=False)

models = []

models.append(RandomForestClassifier())
models.append(skl_da.LinearDiscriminantAnalysis())
models.append(skl_da.QuadraticDiscriminantAnalysis())
models.append(BaggingClassifier())

size = len(models)
print(size)

misclassification = np.zeros((cross, np.shape(models)[0]))

for i in range(cross):
    n = np.ceil(X.shape[0]/cross) # number of samples in each fold
    validationIndex = np.arange(i*n, min(i*n+n, X.shape[0]), 1).astype('int')
    randomize_validationIndex = randomize_indices[validationIndex]
    X_train = X.iloc[~X.index.isin(randomize_validationIndex)]
    Y_train = Y.iloc[~Y.index.isin(randomize_validationIndex)]
    X_validation = X.iloc[randomize_validationIndex]
    Y_validation = Y.iloc[randomize_validationIndex]
    for m in range(np.shape(models)[0]): # try different models
        model = models[m]
        model.fit(X_train, Y_train)
        prediction = model.predict(X_validation)
        misclassification[i, m] = (np.mean(prediction != Y_validation))

plt.boxplot(misclassification)
plt.title('cross validation error for different methods')
plt.xticks(np.arange(size)+1, ('RandomForest', 'LDA', 'QDA', 'Bagging'))
plt.ylabel('validation error')
plt.show()
