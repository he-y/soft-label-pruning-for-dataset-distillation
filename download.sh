# check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found"
    ! pip install gdown
fi

download_all_dataset(){
    # download LPLD distilled dataset
    gdown https://drive.google.com/drive/folders/18qGA3M-3pC6xpAEEFBQ3NGfTHQTz5RfH -O ./recover --folder
}

untar_all_dataset(){
    # untar distilled datasets
    cd recover/syn_data_LPLD
    # untar everything
    for f in *.tar.gz; do tar -xvf $f; done
    cd ../../
}

download_all_labels(){
    # download LPLD labels
    gdown https://drive.google.com/drive/folders/1usMukSUMLe4sx_ZNcKExjOD8G56h6G0L -O ./relabel_and_validate --folder
}

download_40x_labels(){
    DATASET=${1:-"in1k"}
    # check path
    if [ ! -d "./relabel_and_validate/syn_label_LPLD" ]; then
        mkdir -p ./relabel_and_validate/syn_label_LPLD
    fi
    if [ $DATASET == "in1k" ]; then
        # ImageNet-1K [IPC10] 40x
        gdown --fuzzy "https://drive.google.com/file/d/1Nf1piVIXIF-_v-jCEmaYGHdWTXsuQIkY/view?usp=drive_link" -O ./relabel_and_validate/syn_label_LPLD/FKD_cutmix_fp16_LPLD_in1k_rn18_4k_ipc10_ratio975.tar.gz
        # ImageNet-1K [IPC20] 40x
        gdown --fuzzy "https://drive.google.com/file/d/1AdP44DJUadFlY1WCrYiE7F6slotk3Vx4/view?usp=drive_link" -O ./relabel_and_validate/syn_label_LPLD/FKD_cutmix_fp16_LPLD_in1k_rn18_4k_ipc20_ratio975.tar.gz
        # ImageNet-1K [IPC50] 40x
        gdown --fuzzy "https://drive.google.com/file/d/1GnCY-Apg-dXgZe8BvDwDKqrQSAz1PAbs/view?usp=drive_link" -O ./relabel_and_validate/syn_label_LPLD/FKD_cutmix_fp16_LPLD_in1k_rn18_4k_ipc50_ratio975.tar.gz
        # ImageNet-1K [IPC100] 40x
        gdown --fuzzy "https://drive.google.com/file/d/12f6qUjsoN6AczK7iJz2ZAT8xNiX0W4bX/view?usp=drive_link" -O ./relabel_and_validate/syn_label_LPLD/FKD_cutmix_fp16_LPLD_in1k_rn18_4k_ipc100_ratio975.tar.gz
        # ImageNet-1K [IPC200] 40x
        gdown --fuzzy "https://drive.google.com/file/d/1mHWwOaB0yG7fP_lbDSZMmIHUrMh_nDWZ/view?usp=drive_link" -O ./relabel_and_validate/syn_label_LPLD/FKD_cutmix_fp16_LPLD_in1k_rn18_4k_ipc200_ratio975.tar.gz
    elif [ $DATASET == "tiny" ]; then
        # Tiny ImageNet [IPC50] 40x
        gdown --fuzzy "https://drive.google.com/file/d/1Yzgu-I96ODg2J8_AhGuNOP2mlUtbCzHU/view?usp=drive_link" -O ./relabel_and_validate/syn_label_LPLD/FKD_cutmix_fp16_LPLD_tiny_rn18_4k_ipc50_ratio975.tar.gz
        # Tiny ImageNet [IPC100] 40x
        gdown --fuzzy "https://drive.google.com/file/d/1oJuUIq36raTtD63sfzT37ZJ3kGqZGqbv/view?usp=drive_link" -O ./relabel_and_validate/syn_label_LPLD/FKD_cutmix_fp16_LPLD_tiny_rn18_4k_ipc100_ratio975.tar.gz
    elif [ $DATASET == "in21k" ]; then
        # ImageNet-21K [IPC10] 40x
        gdown --fuzzy "https://drive.google.com/file/d/1inuNAC7ApJWiuXaCsEwWU9_z7DOpMBzG/view?usp=drive_link" -O ./relabel_and_validate/syn_label_LPLD/FKD_cutout_fp16_LPLD_in21k_rn18_2k_ipc10_ratio975.tar.gz
        # ImageNet-21K [IPC20] 40x
        gdown --fuzzy "https://drive.google.com/file/d/1g52Lo2XoKHbJySkiLFo3Gsl6hnjffOEN/view?usp=drive_link" -O ./relabel_and_validate/syn_label_LPLD/FKD_cutout_fp16_LPLD_in21k_rn18_2k_ipc20_ratio975.tar.gz
    else
        echo "Invalid dataset name. Please choose from 'in1k', 'tiny', or 'in21k'."
    fi
}

untar_all_labels(){
    # untar labels
    cd relabel_and_validate/syn_label_LPLD
    # untar everything
    for f in *.tar.gz; do tar -xvf $f; done
    cd ../../
}

# usage
download_all_dataset
untar_all_dataset

LESS_LABEL=${1:-"false"}
if [ $LESS_LABEL == "false" ]; then
    echo "Download all labels."
    download_all_labels
else
    echo "Download 40x labels."
    # # optional: download 40x labels
    # # this is much smaller than the original labels
    download_40x_labels in1k
    download_40x_labels tiny
    download_40x_labels in21k
fi
untar_all_labels