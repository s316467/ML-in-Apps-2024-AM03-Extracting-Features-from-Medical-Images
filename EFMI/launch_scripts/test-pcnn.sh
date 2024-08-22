#!/bin/bash

# Variabile che riceve il percorso della cartella DATA
data_dir=${1}
batch_size=${2:-16}
workers=${3:-4}
prefix=${4:-multigpu_b192}
ckptdirprefix=${5:-"./results/pcnn/"}
epochs=${6:-10}
resume=${7:-""}
save_model=${8:-"./results/pcnn/"}
arch=${9:-pdresnet50}
train_split_ratio=${10:-0.8} # percentuale di dati da usare per il train

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

# Creazione delle cartelle per train e test all'interno della cartella DATA
train_dir="$data_dir/train"
test_dir="$data_dir/test"

mkdir -p "$train_dir"
mkdir -p "$test_dir"

# Split data into train and test sets ensuring each patient is represented in both
echo "Splitting data into train and test sets..."

python - <<EOF
import os
import shutil
from sklearn.model_selection import train_test_split

# Directories
root_dir = "${data_dir}"
categories = ['in_roi_patches', 'not_roi_patches']

# Create train and test directories within the original DATA folder
train_dir = "${train_dir}"
test_dir = "${test_dir}"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop over the categories and patients
for category in categories:
    category_dir = os.path.join(root_dir, category)
    print(f"Processing category: {category_dir}")
    
    # Iterate over each patient's directory
    for patient in range(1, 25):  # Assuming patients are labeled from 1 to 24
        patient_dir = os.path.join(category_dir, f"{patient}.svs")
        print(f"Processing patient directory: {patient_dir}")
        
        # Create patient-specific directories in train and test
        train_patient_dir = os.path.join(train_dir, category, f"{patient}.svs")
        test_patient_dir = os.path.join(test_dir, category, f"{patient}.svs")
        
        os.makedirs(train_patient_dir, exist_ok=True)
        os.makedirs(test_patient_dir, exist_ok=True)

        if os.path.isdir(patient_dir):
            files = os.listdir(patient_dir)
            print(f"Found {len(files)} files in {patient_dir}")
            
            if len(files) > 1:
                # Split the files into 80% train and 20% test
                train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
                
                # Copy files to the respective directories within DATA
                for file in train_files:
                    src_file = os.path.join(patient_dir, file)
                    dest_file = os.path.join(train_patient_dir, file)
                    print(f"Copying {src_file} to {dest_file}")
                    shutil.copy2(src_file, dest_file)
                
                for file in test_files:
                    src_file = os.path.join(patient_dir, file)
                    dest_file = os.path.join(test_patient_dir, file)
                    print(f"Copying {src_file} to {dest_file}")
                    shutil.copy2(src_file, dest_file)
            elif len(files) == 1:
                # If only one file, assign it randomly to train or test
                print(f"Only one file found for patient {patient}, assigning to train.")
                src_file = os.path.join(patient_dir, files[0])
                dest_file = os.path.join(train_patient_dir, files[0])
                shutil.copy2(src_file, dest_file)
            else:
                print(f"No files found for patient {patient}.")
        else:
            print(f"Patient directory {patient_dir} does not exist. Creating empty train/test directories for this patient.")

print("Data split complete. Train and test sets are ready.")
EOF

# Costruisci il comando per l'esecuzione
cmd="python ./extractors/pcnn/partialconv/main.py \
-a ${arch} \
--data_train \"${train_dir}\" \
--data_val \"${test_dir}\" \
--batch-size ${batch_size} \
--workers ${workers} \
--prefix ${prefix} \
--ckptdirprefix ${ckptdirprefix} \
--epochs ${epochs} \
--extract-embeddings \
--generate-tsne"

# Aggiungi opzioni opzionali
if [ -n "${resume}" ]; then
    cmd="${cmd} --resume ${resume}"
fi

if [ -n "${save_model}" ]; then
    cmd="${cmd} --save-model \"${save_model}\""
fi

# Esegui il comando
eval $cmd
