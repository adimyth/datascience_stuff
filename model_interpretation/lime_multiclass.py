from __future__ import print_function

import lime
import numpy as np
import sklearn
from lime.lime_text import LimeTextExplainer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

newsgroups_train = fetch_20newsgroups(
    subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(
    subset='test', remove=('headers', 'footers', 'quotes'))
# making class names shorter
class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:])
               for x in newsgroups_train.target_names]
class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

nb = MultinomialNB(alpha=0.01)
nb.fit(train_vectors, newsgroups_train.target)
pred = nb.predict(test_vectors)
print(f1_score(newsgroups_test.target, pred, average='weighted'))

c = make_pipeline(vectorizer, nb)

idx = 100
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(
    newsgroups_test.data[idx], c.predict_proba, num_features=6, labels=[0, 17])
print('Document :', newsgroups_test.data[idx])
print('Predicted class =', class_names[nb.predict(
    test_vectors[idx]).reshape(1, -1)[0, 0]])
print('True class: %s' % class_names[newsgroups_test.target[idx]])

out = exp.as_html()
with open('out.html', 'w') as file:
    file.write(out)
