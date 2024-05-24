# TODO: Code the svm baseline classifier here

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def classify(latent_vectors, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        latent_vectors, labels, test_size=0.2, random_state=42
    )

    # Initialize and train the SVM classifier
    svm_classifier = SVC(kernel="linear", C=1.0)
    svm_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # TODO store these results somewere
