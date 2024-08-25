# TODO: Code the svm baseline classifier here

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils.plotting import *


def apply_pca(train_set, test_set, n_components=128):
    pca = PCA(n_components=n_components)
    train_set_pca = pca.fit_transform(train_set)
    test_set_pca = pca.transform(test_set)

    plot_pca_variance(pca, ".")
    return train_set_pca, test_set_pca


def classify(
    latent_vectors, labels, experiment_name, with_pca=False, pca_components=128
):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        latent_vectors, labels, test_size=0.2, random_state=42
    )

    if with_pca:
        X_train, X_test = apply_pca(X_train, X_test, n_components=pca_components)

    svm_classifier = SVC(kernel="linear", C=1.0)
    print("Fitting svm...")
    svm_classifier.fit(X_train, y_train)
    print("Predicting labels..")
    y_pred = svm_classifier.predict(X_test)

    evaluate(X_test, y_test, y_pred, experiment_name)


def classify_with_provided_splits(
    train_features,
    train_labels,
    test_features,
    test_labels,
    results_path,
    with_pca=True,
    pca_components=128,
):

    if with_pca:
        train_features, test_features = apply_pca(
            train_features, test_features, n_components=pca_components
        )

    # Initialize and train the SVM classifier
    svm_classifier = SVC(kernel="linear", C=1.0)
    print("Fitting svm...")
    svm_classifier.fit(train_features, train_labels)
    print("Predicting labels..")
    y_pred = svm_classifier.predict(test_features)

    evaluate(test_features, test_labels, y_pred, results_path)


def evaluate(test_features, test_labels, y_pred, results_path):
    # Evaluate the classifier
    accuracy = accuracy_score(test_labels, y_pred)
    report = classification_report(test_labels, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

    with open(f"{results_path}.txt", "w") as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write("Classification Report:\n")
        file.write(report)

    print(f"Results saved to {results_path}.txt")

    plot_tsne(
        test_features,
        y_pred,
        title=f"{results_path} t-SNE plot of SVM predictions",
        filename=f"{results_path}_tsne_plot.png",
    )
