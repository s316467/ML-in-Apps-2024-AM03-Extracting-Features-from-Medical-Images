root_dir=${1}
not_roi_path=${2}
roi_path=${3}
patch_size=${4:-512}
mag_level=${5:-1}

export PYTHONPATH=$(pwd)

python ./dataset/stats.py \
--root_dir=${root_dir} \