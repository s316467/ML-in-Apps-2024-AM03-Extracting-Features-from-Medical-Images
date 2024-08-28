root_dir=${1}
num_images=${2:-24}
batch_size=${3:-8}
model_name=${4:-resnet50}
ft_epochs=${5:-10}
latent_dim=${6:-128}


# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

results_path="results/baselines/finetuned/${model_name}_${ft_epochs}_bs${batch_size}_numimages${num_images}_latentdim${latent_dim}"

python ./extractors/baseline/ftmain.py \
--root_dir=${root_dir} \
--num_images=${num_images} \
--batch_size=${batch_size} \
--model_name=${model_name} \
--ft_epochs=${ft_epochs} \
--latent_dim=${latent_dim} \
--results_path=${results_path}