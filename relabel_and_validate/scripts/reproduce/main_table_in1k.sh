cfg=cfg/reproduce/LPLD_in1k_[4k].yaml
gpus="0,1,2,3"

LPLD_imagenet_1k_result(){
    ipc=$1
    for model_size in 18; do
        model=resnet${model_size}

        # 0 -> 1x
        # 0.9 -> 10x
        # 0.95 -> 20x
        # 0.97 -> 30x
        # 0.975 -> 40x
        for ratio in 0 0.9 0.95 0.97 0.975; do
            # pre-defined paths for later generated labels
            fkd_path="syn_label_LPLD/FKD_cutmix_fp16_LPLD_in1k_rn18_4k_ipc${ipc}"
            # fixed to rn18 due to we only use rn18 for recovering
            train_dir="../recover/syn_data_LPLD/LPLD_in1k_rn18_4k_ipc${ipc}"

            # check if the label is already generated
            if [ ! -d $fkd_path ]; then
                python generate_soft_label_pruning.py --cfg_yaml $cfg --fkd_path $fkd_path --train_dir $train_dir --gpus $gpus &
                wait
            fi

            run_name=in1k_rn${model_size}_4k_ipc${ipc}
            output_dir=validate_result/LPLD_in1k_rn${model_size}_[4K]_ipc[${ipc}]

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
                --exp_name LDLP &

        done
    done
}

## example usage:
## LPLD_imagenet_1k_result [IPC]
LPLD_imagenet_1k_result 10