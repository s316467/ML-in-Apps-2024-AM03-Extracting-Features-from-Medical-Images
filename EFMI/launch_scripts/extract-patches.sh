ds_path=${1}
in_roi_path=${2}
not_roi_path=${3}

python ./dataset/patches.py \
--ds_path=${ds_path} \
--in_roi_path=${in_roi_path} \
--not_roi_path=${not_roi_path} \
--patch_size=${patch_size} \
--mag_level=${mag_level} \