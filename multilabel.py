import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
import models

data = pd.read_csv("data.csv", quotechar='"')
X = data.text
y = data[["term1", "term2", "term3"]]

yvalues = y.as_matrix()

final = []
for i in range(0, len(yvalues)):
    r = yvalues[:][i:i + 1]
    r = np.array(r[[~pd.isnull(r)]])
    final.append(r)

final = np.array(final)

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(final)

cv = CountVectorizer(stop_words="english")

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=5)

X_train_dtm = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)
#model = OneVsRestClassifier(SVC(kernel='linear'))
model = models.UniformOVA(c=1, t1=0.3, t2=0.1)
model.fit(X_train_dtm, Y_train)
Y_pred_class = model.predict(X_test_dtm)

score = metrics.accuracy_score(Y_test, Y_pred_class)
print(score)
