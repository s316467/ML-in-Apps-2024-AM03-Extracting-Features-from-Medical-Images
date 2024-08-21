#!/bin/bash

# Variabile che riceve il percorso della cartella DATA
data_dir=${1}
batch_size=${2:-16}
workers=${3:-4}
prefix=${4:-multigpu_b192}
ckptdirprefix=${5:-experiment_1/}
epochs=${6:-1}
resume=${7:-""}
save_model=${8:-""}
arch=${9:-pdresnet50}
train_split_ratio=${10:-0.8} # percentuale di dati da usare per il train

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

# Creazione delle cartelle temporanee per train e test
temp_dir=$(mktemp -d)
train_dir="$temp_dir/train"
test_dir="$temp_dir/test"

mkdir -p "$train_dir"
mkdir -p "$test_dir"

# Funzione per dividere le immagini in train e test
split_data() {
    local src_dir=$1
    local dst_train_dir=$2
    local dst_test_dir=$3
    local ratio=$4
    
    for category in "$src_dir"/*; do
        if [ -d "$category" ]; then
            category_name=$(basename "$category")
            mkdir -p "$dst_train_dir/$category_name"
            mkdir -p "$dst_test_dir/$category_name"
            
            files=("$category"/*)
            total_files=${#files[@]}
            train_count=$((total_files * ratio / 1))  # Calcolo del numero di file per il train set
            
            for ((i=0; i<total_files; i++)); do
                if [ $i -lt $train_count ]; then
                    cp "${files[$i]}" "$dst_train_dir/$category_name/"
                else
                    cp "${files[$i]}" "$dst_test_dir/$category_name/"
                fi
            done
        fi
    done
}

# Esegui lo split per le cartelle in_roi_patches e not_roi_patches
split_data "$data_dir/in_roi_patches" "$train_dir/in_roi_patches" "$test_dir/in_roi_patches" $train_split_ratio
split_data "$data_dir/not_roi_patches" "$train_dir/not_roi_patches" "$test_dir/not_roi_patches" $train_split_ratio

# Costruisci il comando per l'esecuzione
cmd="python ./extractors/pcnn/main.py \
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

# Pulizia delle cartelle temporanee
rm -rf "$temp_dir"
