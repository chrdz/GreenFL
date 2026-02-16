#!/bin/bash
#module load conda/2021.11-python3.9
source ../../venv/bin/activate

### - Parameters to choose for dataset generation - ###
### - Only change here - ###
alpha="0.5" # 0.1:non-iid, 100000:iid, 0: true iid
generate_data=true #true/false
############################

n_tasks="7" # 7 clients, one client per country
#######################################################

###########################
### DATASETS GENERATION ###
###########################
if $generate_data; then
echo "=> generate data"
cd ../..
cd fl_training/data/cifar10 || exit 1
# cd data/cifar10

# --- dataset creation --- #
rm -rf all_data
(
python generate_data.py \
--n_tasks ${n_tasks} \
--s_frac 1.0 \
--test_tasks_frac 0.0 \
--seed 12345 \
--by_labels_split \
--alpha ${alpha}
) # /!\ the two last lines are for non-iid
# --------------------------- #
cd ..
fi


################
### TRAINING ###
################
cd ../../fl_training

echo "=> training"

### - Parameters to choose for training - ###
### - Only change here - ###
availabilities="prob2_CAFE_6cb_1ft_55Rounds prob2_CAFE_7cb_1ft_25Rounds prob2_CAFE_1cb_1ft_150Rounds prob2_CAFE_2cb_1ft_155Rounds prob2_CAFE_3cb_1ft_160Rounds prob2_CAFE_4cb_1ft_145Rounds prob2_CAFE_5cb_1ft_110Rounds prob3_FedZero_6cb_1ft_55Rounds prob3_FedZero_7cb_1ft_25Rounds prob3_FedZero_1cb_1ft_150Rounds prob3_FedZero_2cb_1ft_155Rounds prob3_FedZero_3cb_1ft_160Rounds prob3_FedZero_4cb_1ft_145Rounds prob3_FedZero_5cb_1ft_110Rounds prob1_alphaFair_6cb_1ft_55Rounds prob1_alphaFair_7cb_1ft_25Rounds prob1_alphaFair_1cb_1ft_150Rounds prob1_alphaFair_2cb_1ft_155Rounds prob1_alphaFair_3cb_1ft_160Rounds prob1_alphaFair_4cb_1ft_145Rounds prob1_alphaFair_5cb_1ft_110Rounds" # list of availability matrices
fl_algo="fedavg" # list of FL algorithms
biased="2" # 0:unbiased, 1:biased, 2:hybrid (unbiased except when all clients available)
fine_tuning=1 # Change this to # of finetuning step
grad_clip_threshold="5.0" # Change this to None if you don't want to clip
verbose=0 # 0,1,2
############################

participation="1.0"
heterogeneities="0.0"
weights="0.5" # is the beta parameter in the FedStale paper
seeds="42 78 84"
lrs="1e-2 5e-2" # list of learning rates
device="cuda"
n_rounds="200" # number of fl rounds
#############################################

### other availability matrices' names ###
# opt-new-problem-cvxpy_a-1
# opt-new-problem-cvxpy_a-10
# opt-new-problem-cvxpy_a-21
# uniform-CI-threshold 
# uniform-carbon-budget 
# uniform-carbon-budget-fine-tuning 
# uniform-time-budget
# nonlinear-optimization-cvxpy_w-no-w_a-1 
# nonlinear-optimization-cvxpy_w-no-w_a-0.5 
# random4_uniform-carbon-budget-fine-tuning
# nonlinear-optimization-cvxpy_w-no-w_a-0.1


# ------------------------------ #
# --- Experiments for FedAvg --- #
# ------------------------------ #
# known participation probs:
if echo "$fl_algo" | grep -q "fedavg"; then
# if [[ "fedavg" == *"$fl_algo"* ]]; then
for availability in $availabilities; do
availability_matrix_path="../avMat_baselines_cifar10_bestEndFT/av-mat_${availability}.csv"
if echo "$availability" | grep -qE '[0-9]+Rounds'; then
    n_rounds=$(echo "$availability" | sed -n 's/.*_\([0-9]\+\)Rounds.*/\1/p')
    echo "Number Rounds: $n_rounds"
else
    n_rounds=200
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for seed in $seeds; do
echo "Availability matrix: ${availability} \n Run FedAvg : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
(
python train.py \
cifar10 \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer sgd \
--logs_dir ../logs/cifar10_baselines/${availability}/biased_${biased}/fedavg/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose ${verbose} \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased} \
--fine_tuning ${fine_tuning} \
--grad_clip_threshold ${grad_clip_threshold}
)
done
done
done
done
fi

# ------------------------------- #
# --- Experiments for FedVARP --- #
# ------------------------------- #
# known participation probs:
if echo "$fl_algo" | grep -q "fedvarp"; then
for availability in $availabilities; do
availability_matrix_path="../avMat_baselines_cifar10_bestEndFT/av-mat_${availability}.csv"
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for seed in $seeds; do
echo "Run FedVARP : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
(
python train.py \
cifar10 \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer history \
--logs_dir ../logs/cifar10_debug/${availability}/biased_${biased}/fedvarp/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose ${verbose} \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased} \
--fine_tuning ${fine_tuning} \
--grad_clip_threshold ${grad_clip_threshold}
)
done
done
done
done
fi

# -------------------------------- #
# --- Experiments for FedStale --- #
# -------------------------------- #
# known participation probs:
if echo "$fl_algo" | grep -q "fedstale"; then
for availability in $availabilities; do
availability_matrix_path="../avMat_baselines_cifar10_bestEndFT/av-mat_${availability}.csv"
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for weight in $weights; do
for seed in $seeds ; do
echo "Run FedStale : p ${participation}, h ${heterogeneity}, beta ${weight}, lr ${lr}, seed ${seed}"
(
python train.py \
cifar10 \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer history \
--history_coefficient ${weight} \
--logs_dir ../logs/cifar10_debug/${availability}/biased_${biased}/fedstale/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose ${verbose} \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased} \
--fine_tuning ${fine_tuning} \
--grad_clip_threshold ${grad_clip_threshold}
)
done
done
done
done
done
fi
