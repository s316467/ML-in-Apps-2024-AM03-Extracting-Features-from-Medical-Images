root_dir=${1}
num_images=${2:-24}
batch_size=${3:-16}
latent_dim=${4:-128}
fine_tune_epochs=${5:-10}
results_path=${6:-./results/pathdino}
pretrained_dino_path=${7:-./extractors/pathdino/model/PathDino512.pth}


# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)


python ./extractors/pathdino/main.py \
--root_dir=${root_dir} \
--num_images=${num_images} \
--batch_size=${batch_size} \
--latent_dim=${latent_dim} \
--fine_tune_epochs=${fine_tune_epochs} \
--pretrained_dino_path=${pretrained_dino_path} \
--results_path=${results_path}
