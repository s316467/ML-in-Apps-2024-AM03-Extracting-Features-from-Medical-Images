patches_path=${1}
num_images=${2:-1}
batch_size=${3:-16}
model_name=${4:-resnet50}
latent_dim=${5:-128}
results_path=${6}

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

python ./extractors/baseline/regression-main.py \
--patches_path=${patches_path} \
--num_images=${num_images} \
--batch_size=${batch_size} \
--model_name=${model_name} \
--latent_dim=${latent_dim} \
--results_path=${results_path}
