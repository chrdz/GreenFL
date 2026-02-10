#!/bin/bash
module load conda/2020.11-python3.8
source activate green_fl

### - Parameters to choose for dataset generation - ###
### - Only change here - ###
# alpha="0.1" # 0.1:non-iid, 100000:iid, 0: true iid
alpha=$4
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
--s_frac 0.6 \
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
availabilities=$5 # list of availability matrices
fl_algo="fedavg fedvarp" # list of FL algorithms
biased="2" # 0:unbiased, 1:biased, 2:hybrid (unbiased except when all clients available)
verbose=2 # 0,1,2
############################

participation="1.0"
heterogeneities="0.0"
weights="0.5" # is the beta parameter in the FedStale paper
seeds=$1
batch_size=$2
lrs=$3 # list of learning rates
device="cuda"
n_rounds="100" # number of fl rounds
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

echo "Alpha: ${alpha}"

# ------------------------------ #
# --- Experiments for FedAvg --- #
# ------------------------------ #
# known participation probs:
if echo "$fl_algo" | grep -q "fedavg"; then
# if [[ "fedavg" == *"$fl_algo"* ]]; then
for availability in $availabilities; do
availability_matrix_path="../availability_matrices/av-mat_${availability}.csv"
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for seed in $seeds; do
echo "Availability matrix: ${availability} \n Run FedAvg : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
(
python train.py \
cifar10 \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--bz ${batch_size} \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer sgd \
--logs_dir ../logs/cifar10/${batch_size}/${availability}/biased_${biased}/fedavg/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose ${verbose} \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased}
--model_name resnet
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
availability_matrix_path="../availability_matrices/av-mat_${availability}.csv"
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for seed in $seeds; do
echo "Run FedVARP : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
(
python train.py \
cifar10 \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--bz ${batch_size} \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer history \
--logs_dir ../logs/cifar10/${batch_size}/${availability}/biased_${biased}/fedvarp/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose ${verbose} \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased}
--model_name resnet
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
availability_matrix_path="../availability_matrices/av-mat_${availability}.csv"
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
--bz ${batch_size} \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer history \
--history_coefficient ${weight} \
--logs_dir ../logs/cifar10/${batch_size}/${availability}/biased_${biased}/fedstale/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose ${verbose} \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased}
--model_name resnet
)
done
done
done
done
done
fi
