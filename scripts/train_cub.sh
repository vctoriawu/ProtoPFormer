#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,2

# model=deit_tiny_patch16_224
# model=deit_small_patch16_224
# model=cait_xxs24_224
# batch_size=128
# num_gpus=4
model=$1
batch_size=$2
num_gpus=$3

wandb_mode="online" # one of "online", "offline" or "disabled"
run_name="ProtoPFormer_Hyper-CUB-0"

use_port=2672
seed=1028

# Learning Rate
warmup_lr=1e-4
warmup_epochs=0
min_lr=1e-6
features_lr=1e-4
add_on_layers_lr=3e-3
prototype_vectors_lr=3e-3
last_layer_lr=1e-4
last_layer_global_lr=1e-4
curv_lr=5e-4
visual_alpha_lr=5e-4

# Optimizer & Scheduler
opt=adamw
sched=cosine
decay_epochs=10
decay_rate=0.9
weight_decay=0.05
epochs=200
output_dir=output_cosine/
input_size=224

prototype_activation_function="log"
entailment_coe=0  # entailment loss coefficient
feat_range_type="Sigmoid"   # can be "Tanh" or "Sigmoid"
use_global=True
use_ppc_loss=True   # Whether use PPC loss
last_reserve_num=81 # Number of reserve tokens in the last layer
global_coe=0.5      # Weight of the global branch, 1 - global_coe is for local branch
ppc_cov_thresh=1.   # The covariance thresh of PPC loss
ppc_mean_thresh=2.  # The mean thresh of PPC loss
global_proto_per_class=10   # Number of global prototypes per class
ppc_cov_coe=0.1     # The weight of the PPC_{sigma} Loss
ppc_mean_coe=0.5    # The weight of the PPC_{mu} Loss
dim=192             # The dimension of each prototype

if [ "$model" = "deit_small_patch16_224" ]
then
    reserve_layer_idx=11
elif [ "$model" = "deit_tiny_patch16_224" ]
then
    reserve_layer_idx=11
elif [ "$model" = "cait_xxs24_224" ]
then
    reserve_layer_idx=1
fi

ft=protopformer

for data_set in CUB2011U;
do
    prototype_num=2000
    data_path=/workspace/Hyperbolic_Hierarchical_Protonet_dev/data
    python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=$use_port --use_env main.py \
        --wandb_mode=$wandb_mode \
        --run_name=$run_name \
        --base_architecture=$model \
        --data_set=$data_set \
        --data_path=$data_path \
        --input_size=$input_size \
        --output_dir=$output_dir/$data_set/$model/$run_name/$seed-$lr-$opt-$weight_decay-$epochs-$ft \
        --model=$model \
        --batch_size=$batch_size \
        --seed=$seed \
        --opt=$opt \
        --sched=$sched \
        --warmup-epochs=$warmup_epochs \
        --warmup-lr=$warmup_lr \
        --min_lr=$min_lr \
        --decay-epochs=$decay_epochs \
        --decay-rate=$decay_rate \
        --weight_decay=$weight_decay \
        --epochs=$epochs \
        --finetune=$ft \
        --features_lr=$features_lr \
        --add_on_layers_lr=$add_on_layers_lr \
        --prototype_vectors_lr=$prototype_vectors_lr \
        --last_layer_lr=$last_layer_lr \
        --last_layer_global_lr=$last_layer_global_lr \
        --curv_lr=$curv_lr \
        --visual_alpha_lr=$visual_alpha_lr \
        --prototype_shape $prototype_num $dim 1 1 \
        --reserve_layers $reserve_layer_idx \
        --reserve_token_nums $last_reserve_num \
        --prototype_activation_function=$prototype_activation_function \
        --entailment_coe=$entailment_coe \
        --use_global=$use_global \
        --feat_range_type=$feat_range_type \
        --use_ppc_loss=$use_ppc_loss \
        --ppc_cov_thresh=$ppc_cov_thresh \
        --ppc_mean_thresh=$ppc_mean_thresh \
        --global_coe=$global_coe \
        --global_proto_per_class=$global_proto_per_class \
        --ppc_cov_coe=$ppc_cov_coe \
        --ppc_mean_coe=$ppc_mean_coe
done