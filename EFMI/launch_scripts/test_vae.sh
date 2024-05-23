root_dir=${1}
num_images=${2:-1}
batch_size=${3:-8}
latent_dim=${4:-100}

python ./features-extractors/vae/main.py \
--root_dir=${root_dir} \
--num_images=${num_images} \
--batch_size=${batch_size} \
--latent_dim=${latent_dim}