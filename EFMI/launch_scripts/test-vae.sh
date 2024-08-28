root_dir=${1}
num_images=${2:-24}
batch_size=${3:-8}
latent_dim=${4:-100}
num_epochs=${5:-35}
vae_type=${6:-vae}
no_train=${7:-1}
model_path=${8}

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

results_path="results/vae/${vae_type}_bs${batch_size}_numimages${num_images}_latentdim${latent_dim}"


python ./extractors/vae/main.py \
--root_dir=${root_dir} \
--num_images=${num_images} \
--batch_size=${batch_size} \
--latent_dim=${latent_dim} \
--num_epochs=${num_epochs} \
--vae_type=${vae_type} \
--no_train=${no_train} \
--model_path=${model_path} \
--results_path=${results_path} \

