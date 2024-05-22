import json
import pickle

# Step 1: Parse the JSON file to extract filenames
with open('dataset_test.json', 'r') as f:
    data = json.load(f)

# Extract filenames from the JSON data
filenames = [item[0] for item in data['labels']]

# Step 2: Save the filenames list to a pickle file
with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)

print("filenames.pkl file has been generated successfully.")
