root_dir=${1}
num_images=${2:-24}
batch_size=${3:-16}
latent_dim=${4:-128}
pretrained_dino_path=${5:-./extractors/pathdino/model/PathDino512.pth}
fine_tune=${6:-finetune}
finetune_epochs=${7:-10}

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

results_path="pathdino_bs${batch_size}_numimages${num_images}_latentdim${latent_dim}"

python ./extractors/baseline/main.py \
--root_dir=${root_dir} \
--num_images=${num_images} \
--batch_size=${batch_size} \
--latent_dim=${latent_dim} \
--pretrained_dino_path=&{pretrained_dino_path} \
--fine_tune=${fine_tune} \
--fine_tune_epochs=${fine_tune_epochs} \
--results_path=${results_path}
