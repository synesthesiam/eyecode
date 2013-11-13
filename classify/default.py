import numpy as np, pandas
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import auc_score, explained_variance_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVR

DEFAULT_CLASSIFIERS = {
    "dummy"    : DummyClassifier(),
    "bayes"    : GaussianNB(),
    "tree"     : DecisionTreeClassifier(),
    "forest"   : RandomForestClassifier(),
    "extra"    : ExtraTreesClassifier(),
    #"descent"  : SGDClassifier(),
    "logistic" : LogisticRegression()
}

DEFAULT_REGRESSORS = {
    "dummy"    : DummyRegressor(),
    "tree"     : DecisionTreeRegressor(),
    "extra"    : ExtraTreesRegressor(),
    "linear"   : LinearRegression(),
    "ridge"    : Ridge(),
    "svr"      : SVR(),
    #"adaboost" : AdaBoostRegressor()
}

def feature_importances(frame, columns, label, num_estimators=250,
        random_state=None, regressor=False):
    forest = None

    if regressor:
        forest = ExtraTreesRegressor(n_estimators=num_estimators,
            random_state=random_state)
    else:
        forest = ExtraTreesClassifier(n_estimators=num_estimators,
            random_state=random_state)

    X = frame[columns].values
    y = frame[label].values
    forest.fit(X, y)

    # Extract results
    importances = forest.feature_importances_
    importance_stds = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    rows = [[columns[i], importances[i], importance_stds[i],
        frame[[columns[i], label]].corr().values[0, 1]] for i in indices]

    return pandas.DataFrame(rows, columns=["column", "importance", "importance_std", "label_corr"])

def cross_validation(frame, columns, label, classifiers=DEFAULT_CLASSIFIERS,
        classifier_score = "roc_auc", regressors=DEFAULT_REGRESSORS, regressor=False,
        regressor_score = "r2", random_state=None, cv=10, repeat=1):

    results = []
    X = frame[columns].values
    y = frame[label].values

    for i in range(repeat):
        if regressor:
            for reg_name, reg in regressors.iteritems():
                scores = cross_val_score(reg, X, y, cv=cv, scoring=regressor_score)
                results += [[reg_name, s] for s in scores]
        else:
            for cls_name, cls in classifiers.iteritems():
                scores = cross_val_score(cls, X, y, cv=cv, scoring=classifier_score)
                results += [[cls_name, s] for s in scores]

    return pandas.DataFrame(results, columns=["classifier", "score"])

def cross_val_performance(cross_val):
    cross_val = cross_val[cross_val.classifier != "dummy"]
    return cross_val.score

def one_at_a_time(frame, columns, label, norm=False, **kwargs):
    scores = []
    for col in columns:
        cross_val = cross_validation(frame, [col], label, **kwargs)
        score = cross_val_performance(cross_val).mean()
        scores.append(score)

    scores = np.array(scores)
    if norm:
        scores = scores / float(scores.max())

    return scores
