root_dir=${1}
num_images=${2:-24}
batch_size=${3:-8}
model_name=${4:-resnet50}
latent_dim=${5:-128}


# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

results_path="${model_name}_bs${batch_size}_numimages${num_images}_latentdim${latent_dim}"

python ./extractors/baseline/main.py \
--root_dir=${root_dir} \
--num_images=${num_images} \
--batch_size=${batch_size} \
--model_name=${model_name} \
--latent_dim=${latent_dim} \
--results_path=${results_path}
