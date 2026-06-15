#!/bin/bash
export PYTHONPATH=src/third_party:$PYTHONPATH

GPU=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

CUDA_VISIBLE_DEVICES=$GPU python -m src.main +experiment=scannet/embodiedsplat_train_s1.yaml +output_dir=embodiedsplat_train_s1 checkpointing.load="pretrained/freesplatpp_scannet.ckpt"

mv "train_outputs/embodiedsplat_train_s1/checkpoints/epoch=471-step=50000.ckpt" "train_outputs/embodiedsplat_train_s1/checkpoints/50000.ckpt"

CUDA_VISIBLE_DEVICES=$GPU python -m src.main +experiment=scannet/embodiedsplat_train_s1_2.yaml +output_dir=embodiedsplat_train_s1_2 checkpointing.load="train_outputs/embodiedsplat_train_s1/checkpoints/50000.ckpt"

mv "train_outputs/embodiedsplat_train_s1_2/checkpoints/epoch=471-step=50000.ckpt" "train_outputs/embodiedsplat_train_s1_2/checkpoints/50000.ckpt"