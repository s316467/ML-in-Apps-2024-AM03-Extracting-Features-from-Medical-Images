root_dir=${1}
num_images=${2:-24}
batch_size=${3:-8}
latent_dim=${4:-100}
num_epochs=${5:-50}
vae_type=${6:-vae}

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

results_path="$vae_bs${batch_size}_numimages${num_images}_latentdim${latent_dim}"

python ./extractors/vae/main.py \
--root_dir=${root_dir} \
--num_images=${num_images} \
--batch_size=${batch_size} \
--latent_dim=${latent_dim} \
--num_epochs=${num_epochs} \
--vae_type=${vae_type} \
--results_path=${results_path} \
