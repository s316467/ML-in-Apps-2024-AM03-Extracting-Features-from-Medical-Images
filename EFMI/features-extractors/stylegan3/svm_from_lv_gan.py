import json
import numpy as np
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Parse the JSON file to extract filenames and labels
with open('..\dataset\dataset.json', 'r') as f:
    data = json.load(f)

# Extract filenames and labels from the JSON data
filename_to_label = {item[0]: item[1] for item in data['labels']}

# Step 2: Load the embeddings from the pickle file
with open('latent_vectors.pkl', 'rb') as f:
    embeddings_data = pickle.load(f)

# Assuming 'embeddings_data' is a dictionary containing 'w0', 'w1', 'wt0', 'wt1'
embeddings_w0 = embeddings_data['w0'].cpu().numpy()  # Convert tensors to numpy arrays
embeddings_w1 = embeddings_data['w1'].cpu().numpy()
embeddings_wt0 = embeddings_data['wt0'].cpu().numpy()
embeddings_wt1 = embeddings_data['wt1'].cpu().numpy()

# Step 3: Reshape embeddings
# Each embedding has a shape (4, 16, 512). We need to reshape it to (64, 512) and then concatenate
embeddings_w0 = embeddings_w0.reshape(-1, 512)
embeddings_w1 = embeddings_w1.reshape(-1, 512)
embeddings_wt0 = embeddings_wt0.reshape(-1, 512)
embeddings_wt1 = embeddings_wt1.reshape(-1, 512)

# Concatenate all embeddings along the correct axis
X = np.concatenate((embeddings_w0, embeddings_w1, embeddings_wt0, embeddings_wt1), axis=0)

# Step 4: Prepare the labels
# Load the filenames associated with each embedding
with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

# Ensure the number of filenames matches the number of samples in X
print(f'Number of samples in X: {X.shape[0]}')
print(f'Number of filenames: {len(filenames)}')

# Create labels array based on filenames
labels = np.array([filename_to_label[filename] for filename in filenames])

# Ensure the number of labels matches the number of samples in X
print(f'Number of labels: {len(labels)}')

# Verify label distribution
unique_labels, counts = np.unique(labels, return_counts=True)
print(f'Unique labels: {unique_labels}')
print(f'Counts per label: {counts}')

# Step 5: Train the SVM Classifier
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

# Print the distribution of training and test labels
unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
unique_test_labels, test_counts = np.unique(y_test, return_counts=True)
print(f'Training set label distribution: {dict(zip(unique_train_labels, train_counts))}')
print(f'Test set label distribution: {dict(zip(unique_test_labels, test_counts))}')

# Create an SVM classifier
classifier = svm.SVC(kernel='linear')

# Train the classifier
classifier.fit(X_train, y_train)

# Step 6: Evaluate the classifier
# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)