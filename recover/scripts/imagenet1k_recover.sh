#!/bin/bash

recover(){
    ipc=$1
    class_start=$2
    class_end=$3
    arch_checkpoint=${4:-"model_with_class_bn/resnet18_0.pth"}
    python data_synthesis_class.py \
        --arch-name "resnet18" \
        --ipc $ipc \
        --class-start $class_start \
        --class-end $class_end \
        --arch-checkpoint $arch_checkpoint \
        --exp-name LPLD_in1k_rn18_4k_ipc${ipc} \
        --batch-size $ipc \
        --lr 0.25 \
        --iteration 4000 \
        --l2-scale 0 --tv-l2 0 --r-bn 0.01 \
        --verifier --store-best-images \
        --verifier-arch "resnet18" \
        --syn-data-path syn_data_LPLD
}

# Example usage
# GPUS=0,1,2,3
# CUDA_VISIBLE_DEVICES=$GPUS recover [IPC] [CLASS_START] [CLASS_END]
# CUDA_VISIBLE_DEVICES=$GPUS recover 50 0 1000