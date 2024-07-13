# dataset settings
data_name=coco_precomp
data_path=/remote-home/share/zhaozh/NC_Datasets/data
vocab_path=/remote-home/share/zhaozh/NC_Datasets/vocab

# noise settings
noise_ratio=0.6
noise_file=noise_file_by_caption/coco/coco_precomp_0.6.npy

# loss settings
contrastive_loss=InfoNCE
temp=0.07
lam=0.01
q=0.01

# train settings
gpu=0
warmup_epoch=0
num_epochs=60
lr_update=20
batch_size=128
module_name=SGR
folder_name=coco/r_corr_2

CUDA_VISIBLE_DEVICES=$gpu python  train.py      --gpu $gpu \
                                                --warmup_epoch $warmup_epoch \
                                                --folder_name $folder_name \
                                                --noise_ratio $noise_ratio \
                                                --noise_file $noise_file \
                                                --num_epochs $num_epochs \
                                                --batch_size $batch_size \
                                                --lr_update $lr_update \
                                                --module_name $module_name \
                                                --learning_rate 0.0002 \
                                                --data_name $data_name \
                                                --data_path $data_path \
                                                --vocab_path $vocab_path \
                                                --contrastive_loss $contrastive_loss \
                                                --temp $temp \
                                                --lam $lam \
                                                --q $q \
                    
