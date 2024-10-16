#!/bin/bash

recover(){
    ipc=$1
    start=$2
    end=$3
    arch_checkpoint=${4:-"model_with_class_bn/resnet18_tiny_0.pth"}

    python data_synthesis_tiny_class.py \
        --arch-name "resnet18" \
        --arch-path $arch_checkpoint \
        --exp-name "LDLP_tiny_rn18_4k_ipc${ipc}" \
        --syn-data-path syn_data_LPLD \
        --batch-size 100 \
        --lr 0.1 \
        --r-bn 0.05 \
        --iteration 4000 \
        --store-last-images \
        --bn-hook-type "class_stats_training" \
        --ipc-start $start --ipc-end $end \
        --ipc ${ipc}
}

# Example usage
GPUS=0
# CUDA_VISIBLE_DEVICES=$GPUS recover [IPC] [CLASS_START] [CLASS_END]
CUDA_VISIBLE_DEVICES=$GPUS recover 50 0 2