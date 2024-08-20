import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading embeddings...")
embeddings_with_labels = torch.load('./embeddings/eval_embeddings.pt', map_location=torch.device('cpu'))
print("Embeddings loaded")
features = [embedding.cpu().numpy() for embedding, _ in embeddings_with_labels]
labels = [label.cpu().numpy() for _, label in embeddings_with_labels]




# Appiattisci i batch di features in un unico array
features = np.concatenate(features, axis=0)

# Appiattisci i batch di etichette in  un unico array
labels = np.concatenate(labels, axis=0)

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)



# Dividi i dati in set di addestramento e set di test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Addestra un classificatore SVM utilizzando le features estratte
svm_classifier = SVC(kernel='linear')
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