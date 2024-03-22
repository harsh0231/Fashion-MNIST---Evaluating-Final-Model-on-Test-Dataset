import numpy as np
import pandas as pd
import gzip
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

def showImage(data):
    some_article = data   # Selecting the image.
    some_article_image = some_article.reshape(28, 28) # Reshaping it to get the 28x28 pixels
    plt.imshow(some_article_image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

import numpy as np
import gzip

filePath_train_set = '/cxldata/datasets/project/fashion-mnist/train-images-idx3-ubyte.gz'
filePath_train_label = '/cxldata/datasets/project/fashion-mnist/train-labels-idx1-ubyte.gz'
filePath_test_set = '/cxldata/datasets/project/fashion-mnist/t10k-images-idx3-ubyte.gz'
filePath_test_label = '/cxldata/datasets/project/fashion-mnist/t10k-labels-idx1-ubyte.gz'

with gzip.open(filePath_train_label, 'rb') as trainLbpath:
     trainLabel = np.frombuffer(trainLbpath.read(), dtype=np.uint8, offset=8)

with gzip.open(filePath_train_set, 'rb') as trainSetpath:
     trainSet = np.frombuffer(trainSetpath.read(), dtype=np.uint8, offset=16).reshape(len(trainLabel), 784)

with gzip.open(filePath_test_label, 'rb') as testLbpath:
     testLabel = np.frombuffer(testLbpath.read(), dtype=np.uint8, offset=8)

with gzip.open(filePath_test_set, 'rb') as testSetpath:
     testSet = np.frombuffer(testSetpath.read(), dtype=np.uint8, offset=16).reshape(len(testLabel), 784)

X_train = trainSet
X_test = testSet
y_train = trainLabel
y_test = testLabel

showImage(X_train[0])
print("Label:", y_train[0])

np.random.seed(42)

shuffle_index = np.random.permutation(60000)

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)

rnd_clf.fit(X_train, y_train)

y_train_predict = rnd_clf.predict(X_train[0].reshape(1, -1))

y_train_predict = rnd_clf.predict(X_train)

rnd_accuracy = accuracy_score(y_train, y_train_predict)
rnd_precision = precision_score(y_train, y_train_predict, average='weighted')
rnd_recall = recall_score(y_train, y_train_predict, average='weighted')
rnd_f1_score = f1_score(y_train, y_train_predict, average='weighted')

print("Accuracy:", rnd_accuracy)
print("Precision:", rnd_precision)
print("Recall:", rnd_recall)
print("F1 Score:", rnd_f1_score)

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

def calculate_metrics(model, X_train, y_train):
    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=skfolds, scoring="accuracy")
    precision_scores = cross_val_score(model, X_train, y_train, cv=skfolds, scoring="precision_weighted")
    recall_scores = cross_val_score(model, X_train, y_train, cv=skfolds, scoring="recall_weighted")
    f1_scores = cross_val_score(model, X_train, y_train, cv=skfolds, scoring="f1_weighted")
    return accuracy_scores, precision_scores, recall_scores, f1_scores

log_accuracy_scores, log_precision_scores, log_recall_scores, log_f1_scores = calculate_metrics(log_clf, X_train_scaled, y_train)
rnd_accuracy_scores, rnd_precision_scores, rnd_recall_scores, rnd_f1_scores = calculate_metrics(rnd_clf, X_train, y_train)

log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42, max_iter=1000)

print("Softmax Regression (Logistic Regression - multi_class-multinomial) Metrics:")
print("Accuracy Mean:", log_accuracy_scores.mean(), "Accuracy Std:", log_accuracy_scores.std())
print("Precision Mean:", log_precision_scores.mean(), "Precision Std:", log_precision_scores.std())
print("Recall Mean:", log_recall_scores.mean(), "Recall Std:", log_recall_scores.std())
print("F1 Score Mean:", log_f1_scores.mean(), "F1 Score Std:", log_f1_scores.std())

print("\nRandom Forest Classifier Metrics:")
print("Accuracy Mean:", rnd_accuracy_scores.mean(), "Accuracy Std:", rnd_accuracy_scores.std())
print("Precision Mean:", rnd_precision_scores.mean(), "Precision Std:", rnd_precision_scores.std())
print("Recall Mean:", rnd_recall_scores.mean(), "Recall Std:", rnd_recall_scores.std())
print("F1 Score Mean:", rnd_f1_scores.mean(), "F1 Score Std:", rnd_f1_scores.std())

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
log_cv_recall = recall_score(y_train, y_train_pred, average='weighted')
log_cv_f1_score = f1_score(y_train, y_train_pred, average='weighted')

print("Accuracy:", log_cv_accuracy)
print("Precision:", log_cv_precision)
print("Recall:", log_cv_recall)
print("F1 Score:", log_cv_f1_score)

rnd_clf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)

rnd_cv_scores = cross_val_score(rnd_clf, X_train, y_train, cv=3, scoring="accuracy")
display_scores(rnd_cv_scores)

rnd_cv_accuracy = rnd_cv_scores.mean()

y_train_pred = cross_val_predict(rnd_clf, X_train, y_train, cv=3)

conf_matrix = confusion_matrix(y_train, y_train_pred)

rnd_cv_precision = precision_score(y_train, y_train_pred, average='weighted')
rnd_cv_recall = recall_score(y_train, y_train_pred, average='weighted')
rnd_cv_f1_score = f1_score(y_train, y_train_pred, average='weighted')

print("Accuracy:", rnd_cv_accuracy)
print("Precision:", rnd_cv_precision)
print("Recall:", rnd_cv_recall)
print("F1 Score:", rnd_cv_f1_score)

from sklearn.decomposition import PCA

pca = PCA(n_components=0.99)
X_train_reduced = pca.fit_transform(X_train)

print("Number of components:", pca.n_components_)
print("Explained variance ratio:", np.sum(pca.explained_variance_ratio_))

X_train_recovered = pca.inverse_transform(X_train_reduced)

def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.title("Original", fontsize=16)
plt.subplot(122)
plot_digits(X_train_recovered[::2100])
plt.title("Compressed", fontsize=16)
plt.show()

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        "lr__multi_class": ["multinomial"],
        "lr__solver": ["lbfgs"],
        "lr__C": [5],
        "rf__n_estimators": [20],
        "rf__max_depth": [10, 15],
    }]

log_clf_ens = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42)
rnd_clf_ens = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
voting_clf_grid_search = VotingClassifier(
    estimators=[('lr', log_clf_ens), ('rf', rnd_clf_ens)],
    voting='soft')

grid_search = GridSearchCV(voting_clf_grid_search, param_grid, cv=3, scoring='neg_mean_squared_error')

grid_search.fit(X_train_reduced, y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Best estimator:", grid_search.best_estimator_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

final_model = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

X_test_reduced = pca.transform(X_test)

y_test_predict = final_model.predict(X_test_reduced)

confusion_matrix(y_test, y_test_predict)

final_accuracy = accuracy_score(y_test, y_test_predict)
final_precision = precision_score(y_test, y_test_predict, average='weighted')
final_recall = recall_score(y_test, y_test_predict, average='weighted')
final_f1_score = f1_score(y_test, y_test_predict, average='weighted')

print("Final Accuracy:", final_accuracy)
print("Final Precision:", final_precision)
print("Final Recall:", final_recall)
print("Final F1 Score:", final_f1_score)

print("Actual Label:", y_test[0])
print("Predicted Label:", y_test_predict[0])
showImage(X_test[0])
