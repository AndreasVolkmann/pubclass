from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np


data = pd.read_csv("pub-learn.csv", quotechar='"')
X = data.abstract
y = data[data.columns[3:]]

yvalues = y.as_matrix()

final = []
for i in range(0, len(yvalues)):
    r = yvalues[:][i:i+1]
    r = np.array(r[[~pd.isnull(r)]])
    final.append(r)

final = np.array(final)


mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(final)
print(X.shape)
print(Y.shape)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english")

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

X_train_dtm = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)
ovr = OneVsRestClassifier(SVC(kernel='linear'))
ovr.fit(X_train_dtm, Y_train)
print("Multilabel: ", ovr.multilabel_)
Y_pred_class = ovr.predict(X_test_dtm)


from sklearn import metrics
score = metrics.accuracy_score(Y_test, Y_pred_class)
print(score)
