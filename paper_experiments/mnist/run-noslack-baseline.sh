#!/bin/bash
#module load conda/2021.11-python3.9
source ../../greenfl_venv_oldpytorch/bin/activate

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
availabilities="alphaF50-0.0cb-3ft" # list of availability matrices
# availabilities="alphaF50-41sl-2cb-3ft alphaF50-42sl-2cb-3ft alphaF50-43sl-2cb-3ft alphaF50-44sl-2cb-3ft alphaF50-45sl-2cb-3ft alphaF50-46sl-2cb-3ft alphaF50-47sl-2cb-3ft alphaF50-48sl-2cb-3ft alphaF50-49sl-2cb-3ft alphaF50-31sl-4cb-3ft alphaF50-32sl-4cb-3ft alphaF50-33sl-4cb-3ft alphaF50-34sl-4cb-3ft alphaF50-35sl-4cb-3ft alphaF50-36sl-4cb-3ft alphaF50-37sl-4cb-3ft alphaF50-38sl-4cb-3ft alphaF50-39sl-4cb-3ft alphaF50-40sl-4cb-3ft alphaF50-21sl-6cb-3ft alphaF50-22sl-6cb-3ft alphaF50-23sl-6cb-3ft alphaF50-24sl-6cb-3ft alphaF50-25sl-6cb-3ft alphaF50-26sl-6cb-3ft alphaF50-27sl-6cb-3ft alphaF50-28sl-6cb-3ft alphaF50-29sl-6cb-3ft alphaF50-30sl-6cb-3ft alphaF50-40sl-6cb-3ft alphaF50-11sl-8cb-3ft alphaF50-12sl-8cb-3ft alphaF50-13sl-8cb-3ft alphaF50-14sl-8cb-3ft alphaF50-15sl-8cb-3ft alphaF50-16sl-8cb-3ft alphaF50-17sl-8cb-3ft alphaF50-18sl-8cb-3ft alphaF50-19sl-8cb-3ft alphaF50-20sl-8cb-3ft alphaF50-30sl-8cb-3ft alphaF50-40sl-8cb-3ft alphaF50-10sl-9cb-3ft alphaF50-11sl-9cb-3ft alphaF50-12sl-9cb-3ft alphaF50-13sl-9cb-3ft alphaF50-14sl-9cb-3ft alphaF50-15sl-9cb-3ft alphaF50-16sl-9cb-3ft alphaF50-17sl-9cb-3ft alphaF50-18sl-9cb-3ft alphaF50-19sl-9cb-3ft alphaF50-29sl-9cb-3ft alphaF50-39sl-9cb-3ft alphaF50-6sl-10cb-3ft alphaF50-7sl-10cb-3ft alphaF50-8sl-10cb-3ft alphaF50-9sl-10cb-3ft alphaF50-10sl-10cb-3ft alphaF50-11sl-10cb-3ft alphaF50-12sl-10cb-3ft alphaF50-13sl-10cb-3ft alphaF50-14sl-10cb-3ft alphaF50-15sl-10cb-3ft alphaF50-25sl-10cb-3ft alphaF50-35sl-10cb-3ft alphaF50-45sl-10cb-3ft alphaF50-5sl-11cb-3ft alphaF50-6sl-11cb-3ft alphaF50-7sl-11cb-3ft alphaF50-8sl-11cb-3ft alphaF50-9sl-11cb-3ft alphaF50-10sl-11cb-3ft alphaF50-11sl-11cb-3ft alphaF50-12sl-11cb-3ft alphaF50-13sl-11cb-3ft alphaF50-14sl-11cb-3ft alphaF50-24sl-11cb-3ft alphaF50-34sl-11cb-3ft alphaF50-44sl-11cb-3ft" # list of availability matrices
#alphaF50-199sl-1cb-3ft alphaF50-199sl-2cb-3ft alphaF50-197sl-3cb-3ft alphaF50-197sl-4cb-3ft alphaF50-197sl-5cb-3ft alphaF50-197sl-6cb-3ft alphaF50-196sl-7cb-3ft alphaF50-197sl-8cb-3ft alphaF50-197sl-9cb-3ft alphaF50-197sl-10cb-3ft alphaF50-196sl-11cb-3ft
fl_algo="fedavg" # list of FL algorithms
biased="2" # 0:unbiased, 1:biased, 2:hybrid (unbiased except when all clients available)
fine_tuning=3 # Change this to # of finetuning step
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
    n_rounds=50
fi


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
    n_rounds=50
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
--logs_dir ../logs/mnist_baseline/${availability}/biased_${biased}/fedavg/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose ${verbose} \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased} \
--fine_tuning ${fine_tuning} \
--grad_clip_threshold ${grad_clip_threshold} \
--chkpts_dir ../baseline_chkpt/mnist
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
--logs_dir ../logs/mnist_debug/${availability}/biased_${biased}/fedvarp/alpha_${alpha}/lr_${lr}/seed_${seed} \
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
--logs_dir ../logs/mnist_debug/${availability}/biased_${biased}/fedstale/alpha_${alpha}/lr_${lr}/seed_${seed} \
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
