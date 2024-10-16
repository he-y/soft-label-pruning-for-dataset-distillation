wandb disabled

recover_in21k_use_sre2l_config(){
    ipc=$1
    class_start=$2
    class_end=$3
    python data_synthesis_in21k_class.py \
        --arch-name "resnet18" \
        --sre2l \
        --bn-hook-type class_stats_training \
        --arch-checkpoint model_with_class_bn/resnet18_21k_0.pth \
        --exp-name "LPLD_in21k_rn18_2k_ipc${ipc}" \
        --syn-data-path './syn_data_LPLD' \
        --batch-size $ipc \
        --lr 0.05 \
        --r-bn 0.25 \
        --iteration 2000 \
        --store-best-images \
        --ipc $ipc \
        --class-start $class_start --class-end $class_end
}

# Usage: recover_in21k_use_sre2l_config [IPC] [CLASS_START] [CLASS_END]
gpu=0
CUDA_VISIBLE_DEVICES=$gpu recover_in21k_use_sre2l_config 10 0 10450