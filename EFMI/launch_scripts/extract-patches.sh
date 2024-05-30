ds_path=${1}
not_roi_path=${2}
roi_path=${3}
patch_size=${4:-512}
mag_level=${5:-1}

export PYTHONPATH=$(pwd)

python ./dataset/patches.py \
--ds_path=${ds_path} \
--not_roi_path=${not_roi_path} \
--roi_path=${roi_path} \
--patch_size=${patch_size} \
--mag_level=${mag_level}