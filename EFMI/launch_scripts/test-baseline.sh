root_dir=${1}
num_images=${2:-24}
batch_size=${3:-8}
model=${4:-resnet50}
latent_dim=${5:-512}

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

experiment_name="${model}_bs${batch_size}_numimages${num_images}_latentdim${latent_dim}"

python ./baseline/main.py \
--root_dir=${root_dir} \
--num_images=${num_images} \
--batch_size=${batch_size} \
--model=${model} \
--latent_dim=${latent_dim} \
--experiment_name=${experiment_name}
