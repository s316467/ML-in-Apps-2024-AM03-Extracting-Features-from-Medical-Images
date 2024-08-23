ds_path=${1}
wsi_paths=${2}
xml_paths=${3}
patch_size=${4:-512}
mag_level=${5:-1}

export PYTHONPATH=$(pwd)

python ./dataset/regression-main.py \
--ds_path=${ds_path} \
--wsi_paths=${wsi_paths} \
--xml_paths=${xml_paths} \
--patch_size=${patch_size} \
--mag_level=${mag_level}