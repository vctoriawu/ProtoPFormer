#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=0

####################  lessons learned overall
### 1. large LR collapses! 5e-5 for the hyperbolic and last layer stuff is good!
### 2. large entailment coefficient collapses. 0 also collapses.  0.01 and 0.1 are good
### 3. large Batch_size slows down, and collapses too
### 4. warmup slows down the learning, but also ends lower!
### 5 some runs need  more epochs to finish training (still going up). maybe that + warmup?
#     Warmup was worse! more epochs didn't help! Maybe modify the LR scheduler to go lower (so more time in low)
#     or change the scheduler

# NAN
#run_name="ProtoPFormer_Hyper-CUB-fusedLastLayer-NoSigmoid_addOnLayers"

# Collapse
#run_name="ProtoPFormer_Hyper-CUB-fusedLastLayer"

# not much gain! bad cluster and sep!
#run_name="ProtoPFormer_Hyper-CUB-fusedLastLayer-Ent0-warmup"

# fixed the cluster and sep of local prots!

####################################################################################################
######################################  with entailment ########################################
####################################################################################################
#sched=cosine

# RESULT :    COLLAPSED AGAIN
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep"
#######################################  Warm up  # RESULT   : Collapsed too! but after warmup was done!
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep_warmup"

#######################################  lower LR and entailment
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent02-5e5LR"  # RESULT: Collapsed
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent02-linear-5e5LR"   # RESULT: Collapsed later at 20%!
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent02-linear-5e5LR-B128"   # RESULT: Collapsed later at 26%!
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent01-5e5LR"  # RESULT: Collapsed
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent01-linear-5e5LR"  # ****** BEST RESULT  75.99% ***** more epochs didn't help
run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent01-linear-5e5LR-StepLR_08"   # TODO CHECK IF NAN or sta
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent01-linear-5e5LR-noSigmoid"  # NAN
####### Tanh
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent01-linear-5e5LR-tanh_randn"  # Result: 74.3% but can go higher!
run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent02-linear-5e5LR-tanh_randn"  # TODO CHECK IF NAN or stable?
######## lower minLR to cooldown more?
run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent02-linear-5e5LR-tanh_randn-min_LR_1e7"  # TODO CHECK IF NAN or stable?
######## Step Learning rate scheduler
run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent02-linear-5e5LR-tanh_randn-StepLR_08"  # TODO CHECK IF NAN or stable?

#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent01-linear-5e5LR-300e"  # 75.56%
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent01-linear-5e5LR-300e-warmup"  # 75.47%
##run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent01-linear-5e5LR-B256"   # meh, not so good! 70%
##run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent001-linear-5e5LR"   # ****** REALLY GOOD RESULT   75.88% ****** more epochs didn't help
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent001-linear-5e5LR-300e"  # 75.89%
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent001-linear-5e5LR-300e-warmup"  #  75.87%
##run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent001-linear-5e5LR-B256"   # COLLAPSED

####################################################################################################
#######################################  ent 0 ########################################
####################################################################################################
#entailment_coe=0  # entailment loss coefficient
########################################  # results were meh! warmup was worse! maybe it's  not enough LR
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent0"
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent0-warmup"
#
########################################  log vs linear
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent0-linear"  # OK. 74.8%
#run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent0-linear-5e5LR"  # RESULT: much better than above  75.7%
#
########################################  log, but with larger LR
##run_name="ProtoPFormer_Hyper-CUB-FixedClstrSep-Ent0-3e3LR"  # Collapsed again! so maybe collapse = large LR
#
########################################  add_on_layers, 2 vs 1 Conv2D
#
########################################  larger batch size!

# model=deit_tiny_patch16_224
# model=deit_small_patch16_224
# model=cait_xxs24_224
# batch_size=128
# num_gpus=4

#model=$1
#batch_size=$2
#num_gpus=$3

model=deit_small_patch16_224
batch_size=64
num_gpus=1

wandb_mode="online" # one of "online", "offline" or "disabled"

use_port=$((2672 + $CUDA_VISIBLE_DEVICES))
seed=1028

# Learning Rate
warmup_lr=1e-4
warmup_epochs=0  # TODO TRY 5
min_lr=1e-6
features_lr=1e-4
add_on_layers_lr=3e-3
prototype_vectors_lr=5e-4
last_layer_lr=5e-5
last_layer_global_lr=5e-5
curv_lr=5e-5
visual_alpha_lr=5e-5

# Optimizer & Scheduler
opt=adamw
sched=cosine
#sched=step
decay_epochs=10
decay_rate=0.9
#decay_rate=0.8
weight_decay=0.05
epochs=400
output_dir=output_cosine/
input_size=224

entailment_coe=0.2  # entailment loss coefficient
prototype_activation_function="linear"
feat_range_type="Sigmoid"   # can be "Tanh" or "Sigmoid"
use_global=True
use_ppc_loss=False   # Whether use PPC loss  # TODO TRY TRUE?
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