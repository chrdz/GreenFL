#!/bin/bash
#module load conda/2021.11-python3.9
source ../../venv/bin/activate

### - Parameters to choose for dataset generation - ###
### - Only change here - ###
alpha="0.1" # 0.1:non-iid, 100000:iid, 0: true iid
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
cd fl_training/data/mnist || exit 1
# cd data/mnist

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
availabilities="gp100-50-tusu-10ft gp100 gp100-50-tcsc-10ft gp100-50-tcsu-10ft gp100-50-tusc-10ft gp100-35-tusu-10ft gp100-35-tcsc-10ft gp100-35-tcsu-10ft gp100-35-tusc-10ft gp100-25255050507575-tcsu-10ft gp100-25255050507575-tusu-10ft gp100-20304050607080-tcsu-10ft gp100-20304050607080-tusu-10ft" # list of availability matrices
# gp50-50-tusu-3ft gp50-35-tusu-3ft DONE
fl_algo="fedavg" # list of FL algorithms
biased="2" # 0:unbiased, 1:biased, 2:hybrid (unbiased except when all clients available)
fine_tuning=10 # Change this to # of finetuning step
grad_clip_threshold="1.0" # Change this to None if you don't want to clip
verbose=0 # 0,1,2
############################

participation="1.0"
heterogeneities="0.0"
weights="0.5" # is the beta parameter in the FedStale paper
seeds="42"
lrs="5e-2" # list of learning rates
device="cuda"
#n_rounds="100" # number of fl rounds
#############################################
if echo "$availability" | grep -qE '[0-9]+sl'; then
    n_rounds=$(echo "$availability" | sed -n 's/.*-\([0-9]\+\)sl.*/\1/p')
else
    n_rounds=100
fi

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
availability_matrix_path="../availability_matrices/av-mat_${availability}.csv"
#n_rounds=$(echo "$availability" | sed -n 's/.*-\([^-]*\)sl.*/\1/p')
if echo "$availability" | grep -qE '[0-9]+sl'; then
    n_rounds=$(echo "$availability" | sed -n 's/.*-\([0-9]\+\)sl.*/\1/p')
else
    n_rounds=100
fi
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for seed in $seeds; do
echo "Availability matrix: ${availability} \n Run FedAvg : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
(
python train.py \
mnist \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer sgd \
--logs_dir ../logs/mnist_sp/${availability}/biased_${biased}/fedavg/alpha_${alpha}/lr_${lr}/seed_${seed} \
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

# rsync -avu ../../logs/mnist_sp/alphaF50* ../../logs/mnist_server/128/
# ------------------------------- #
# --- Experiments for FedVARP --- #
# ------------------------------- #
# known participation probs:
if echo "$fl_algo" | grep -q "fedvarp"; then
for availability in $availabilities; do
availability_matrix_path="../availability_matrices/av-mat_${availability}.csv"
n_rounds=$(echo "$availability" | sed -n 's/.*-\([^-]*\)sl.*/\1/p')
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for seed in $seeds; do
echo "Run FedVARP : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
(
python train.py \
mnist \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer history \
--logs_dir ../logs/mnist_sp/${availability}/biased_${biased}/fedvarp/alpha_${alpha}/lr_${lr}/seed_${seed} \
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
availability_matrix_path="../availability_matrices/av-mat_${availability}.csv"
n_rounds=$(echo "$availability" | sed -n 's/.*-\([^-]*\)sl.*/\1/p')
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for weight in $weights; do
for seed in $seeds ; do
echo "Run FedStale : p ${participation}, h ${heterogeneity}, beta ${weight}, lr ${lr}, seed ${seed}"
(
python train.py \
mnist \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer history \
--history_coefficient ${weight} \
--logs_dir ../logs/mnist_sp/${availability}/biased_${biased}/fedstale/alpha_${alpha}/lr_${lr}/seed_${seed} \
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
