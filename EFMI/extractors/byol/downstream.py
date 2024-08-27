import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.plotting import plot_tsne
import pickle


def load_features(file_path):
    with open(file_path, "rb") as f:
        train_X, train_y, test_X, test_y = pickle.load(f)
    print(f"Features loaded from {file_path}")
    return train_X, train_y, test_X, test_y


# Dividi i dati in set di addestramento e set di test
X_train, y_train, X_test, y_test = load_features(
    "./extractors/byol/embeddings/eval_embeddings.pt"
)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

plot_tsne(X_train, y_train, "BYOL", "BYOL_train_tsne.png")
plot_tsne(X_test, y_test, "BYOL", "BYOL_test_tsne.png")


# Addestra un classificatore SVM utilizzando le features estratte
svm_classifier = SVC(kernel="linear")
svm_classifier.fit(X_train, y_train)

# Valuta le prestazioni del modello sul set di addestramento
y_pred_train = svm_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print("Accuracy on training set:", train_accuracy)
print("Classification Report on training set:")
print(classification_report(y_train, y_pred_train))
print("Confusion Matrix on training set:")
print(confusion_matrix(y_train, y_pred_train))

# Valuta le prestazioni del modello sul set di test
y_pred_test = svm_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("\nAccuracy on test set:", test_accuracy)
print("Classification Report on test set:")
print(classification_report(y_test, y_pred_test))
print("Confusion Matrix on test set:")
print(confusion_matrix(y_test, y_pred_test))


# create tsne
