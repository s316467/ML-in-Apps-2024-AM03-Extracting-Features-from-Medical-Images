ds_path=${1}
patches_path=${2}
patch_size=${3:-512}
mag_level=${4:-1}

export PYTHONPATH=$(pwd)

python ./dataset/regression-patches.py \
--ds_path=${ds_path} \
--patches_path=${patches_path} \
--patch_size=${patch_size} \
--mag_level=${mag_level}