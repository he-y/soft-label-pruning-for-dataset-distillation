cfg=cfg/reproduce/LPLD_tiny_[4k].yaml
# NOTE: very important for tiny-imagenet to NOT use multiple GPUs
gpus=0

LPLD_tiny_imagenet_result(){
    ipc=$1
    for model_size in 18; do
        model=resnet${model_size}

        # 0 -> 1x
        # 0.9 -> 10x
        # 0.95 -> 20x
        # 0.97 -> 30x
        # 0.975 -> 40x
        for ratio in 0.9 0.95 0.97 0.975; do
            # pre-defined paths for later generated labels
            ratio_path=$(echo $ratio | sed 's/0\.//g')
            fkd_path="syn_label_LPLD/FKD_cutmix_fp16_LPLD_tiny_rn18_4k_ipc${ipc}_ratio${ratio_path}"
            train_dir="../recover/syn_data_LPLD/LPLD_tiny_rn18_4k_ipc${ipc}"

            teacher_ckpt="../recover/tiny_forward/resnet18_tiny_ep50.pth"

            # check if the label is already generated
            if [ ! -d $fkd_path ]; then
                # teacher_ckpt is required for tiny-imagenet (as no pre-trained model is provided in pytorch)
                python generate_soft_label_pruning.py --cfg_yaml $cfg --teacher_ckpt $teacher_ckpt --fkd_path $fkd_path --train_dir $train_dir --gpus $gpus \
                    --prune_ratio $ratio
            fi

            run_name=tiny_rn${model_size}_4k_ipc${ipc}
            output_dir=validate_result/LPLD_tiny_rn${model_size}_[4K]_ipc[${ipc}]

            prune_metric=random
            prune_granularity=batch_to_epoch
            # do not use random sampling for 0.0 ratio, as it keeps all the labels
            if [ $ratio == 0.0 ]; then
                prune_metric=order
                prune_granularity=epoch
            fi

            python train_FKD_label_pruning_batch.py \
                --model ${model} \
                --prune_ratio ${ratio} \
                --granularity $prune_granularity \
                --prune_metric $prune_metric \
                --gpus $gpus \
                --cfg_yaml $cfg \
                --fkd_path ${fkd_path} \
                --train_dir ${train_dir} \
                --output_dir ${output_dir} \
                --run_name ${run_name} \
                --exp_name LDLP 

        done
    done
}

## example usage:
## LPLD_tiny_imagenet_result [IPC]
LPLD_tiny_imagenet_result 50