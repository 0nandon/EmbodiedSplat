#!/bin/bash
export PYTHONPATH=src/third_party:$PYTHONPATH

GPU=3

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

CUDA_VISIBLE_DEVICES=$GPU python -m src.main +experiment=scannetpp/embodiedsplat_fast_test.yaml +output_dir='' mode=test dataset/view_sampler=evaluation checkpointing.load=pretrained/freesplatpp_scannetpp.ckpt dataset.view_sampler.num_context_views=103 model.encoder.num_views=30
