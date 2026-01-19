train_config_name=$1
model_name=$2
gpu_use=$3
export OPENPI_DATA_HOME="/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/gezuhao/AdaMoE-VLA/ckpt/openpi"
export HF_LEROBOT_HOME="/inspire/hdd/project/wuliqifa/public/gezuhao"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=$gpu_use
export XDG_CACHE_HOME="/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/gezuhao/AdaMoE-VLA/cache/xdg"
echo $CUDA_VISIBLE_DEVICES
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python scripts/train.py $train_config_name --exp-name=$model_name --overwrite
