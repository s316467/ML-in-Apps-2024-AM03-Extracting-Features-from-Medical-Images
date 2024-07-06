# TODO: Code the svm baseline classifier here

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils.plotting import * 

from utils.plotting import plot_pca_variance


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

    # Initialize and train the SVM classifier
    svm_classifier = SVC(kernel="linear", C=1.0)
    svm_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

    with open(f"./{experiment_name}.txt", "w") as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write("Classification Report:\n")
        file.write(report)

    print(f"Results saved to {experiment_name}.txt")

    plot_tsne(
        X_test,
        y_pred,
        title="t-SNE plot of SVM predictions",
        filename="tsne_plot.png",
    )
