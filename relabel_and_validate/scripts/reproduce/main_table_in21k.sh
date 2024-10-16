cfg=cfg/reproduce/LPLD_in21k_[2k].yaml
class_forward(){
    ipc=$1
    gpus="0,1,2,3"
    for model_size in 18; do
        model=resnet${model_size}
        for ratio in 0.975 0.97 0.95 0.9 0.0; do
            fkd_path="syn_label_LPLD/FKD_cutout_fp16_LPLD_in21k_rn18_2k_ipc${ipc}"
            train_dir="../recover/syn_data_LPLD/LPLD_in21k_rn18_2k_ipc${ipc}"

            run_name=LPLD_in21k_rn${model_size}_2k_ipc${ipc}
            output_dir=validate_result/LPLD_in21k_rn${model_size}_[2K]_ipc[${ipc}]

            if [ ! -d $fkd_path ]; then
                python generate_soft_label_pruning.py --cfg_yaml $cfg --fkd_path $fkd_path --train_dir $train_dir --gpus $gpus &
            fi

            python train_FKD_label_pruning_batch.py \
                --model ${model} \
                --prune_ratio ${ratio} \
                --granularity batch_to_epoch \
                --prune_metric random \
                --gpus $gpus \
                --cfg_yaml $cfg \
                --fkd_path ${fkd_path} \
                --train_dir ${train_dir} \
                --output_dir ${output_dir} \
                --val_interval 50 \
                --run_name ${run_name} \
                --exp_name label_pruning &

        done
    done
}

# pip install timm
echo "Checking if timm is installed"
if ! python -c "import timm"; then
    echo "timm is not installed, installing now"
    pip install timm
fi

# basic usage:
# class_forward [IPC]
class_forward 10
class_forward 20