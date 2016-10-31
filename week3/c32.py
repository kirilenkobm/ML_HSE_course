import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn import grid_search

# Loading data from 20_newsgroups set from categories 'space' and 'atheism'
newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space'])


# def for extract the most important words for every set
def most_informative_feature_for_class_svm(vectorizer, classifier,  n=10):
    labelid = 0 # this is the coef we're interested in.
    feature_names = vectorizer.get_feature_names()
    svm_coef = classifier.coef_.toarray()
    topn = sorted(zip(svm_coef[labelid], feature_names))[-n:]

    for coef, feat in topn:
        print(feat, coef)
    down = sorted(zip(svm_coef[labelid], feature_names))[:n]
    for coef, feat in down:
        print(feat, coef)


# TF-IDF compilation of text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data, newsgroups.target)
y = newsgroups.target

# Find the best C parameter
if False:
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
    clf = svm.SVC(kernel='linear', random_state=241)
    gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(X, y)
    for a in gs.grid_scores_:
        print(a)
        print(a.mean_validation_score)
        print(a.parameters)
        # best C=10000

# Initiation and training of SVC classifier
clf = svm.SVC(C=10000, kernel='linear', random_state=241)
clf.fit(X, y)
most_informative_feature_for_class_svm(vectorizer, clf)
