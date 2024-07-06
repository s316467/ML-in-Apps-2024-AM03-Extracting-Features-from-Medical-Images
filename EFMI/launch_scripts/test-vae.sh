root_dir=${1}
num_images=${2:-24}
batch_size=${3:-8}
latent_dim=${4:-100}

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

results_path="$vae_bs${batch_size}_numimages${num_images}_latentdim${latent_dim}"

python ./extractors/vae/main.py \
--root_dir=${root_dir} \
--num_images=${num_images} \
--batch_size=${batch_size} \
--latent_dim=${latent_dim}
--results_path=${results_path}
