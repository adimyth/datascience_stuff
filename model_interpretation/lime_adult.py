from __future__ import print_function

import warnings

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.ensemble
import xgboost
from lime.lime_tabular import LimeTabularExplainer

np.random.seed(1)
warnings.filterwarnings('ignore')


data = np.genfromtxt('data/adult.data', delimiter=', ', dtype=str)
feature_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
                 "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"]

labels = data[:, 14]
le = sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:, :-1]

categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_

data = data.astype(float)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
    data, labels, train_size=0.80)

encoder = sklearn.preprocessing.OneHotEncoder(
    categorical_features=categorical_features)
encoder.fit(data)
encoded_train = encoder.transform(train)

gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(encoded_train, labels_train)

print(sklearn.metrics.accuracy_score(
    labels_test, gbtree.predict(encoder.transform(test))))


def predict_fn(x): return gbtree.predict_proba(
    encoder.transform(x)).astype(float)


explainer = LimeTabularExplainer(train, feature_names=feature_names, class_names=class_names,
                                 categorical_features=categorical_features, categorical_names=categorical_names, kernel_width=3)

i = 1653
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
out = exp.as_html()
with open('out.html', 'w') as file:
    file.write(out)
