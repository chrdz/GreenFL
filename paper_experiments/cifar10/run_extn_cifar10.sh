#!/bin/bash
source ../../greenfl_venv/bin/activate

#######################################################
### - Parameters to choose for dataset generation - ###
#######################################################
# Choose alpha between 0 and 1 to determine the level of non-iid ness of the clients datasets
# This is not the alpha-fairness parameter
alpha="0.5" # distribution of data among clients: 0.1:non-iid, 100000:iid, 0: true iid
generate_data=true #true/false true will regenerate the clients' datasets
#######################################################


###########################
### DATASETS GENERATION ###
###########################
n_tasks="7" # 7 clients, one client per country

if $generate_data; then
echo "=> generate data"
cd ../..
cd fl_training/data/cifar10 || exit 1

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
###########################


#############################################
### - Parameters to choose for training - ###
#############################################

logs_folder="cifar10_group_notclip"

### AVAILABILITY MATRIX ###
# Which availability matrix/matrices are you using?
# availabilities="alphaF-0cb-3ft alphaF-120sl-1cb-3ft alphaF-120sl-6cb-3ft alphaF-120sl-3cb-3ft alphaF-120sl-5cb-3ft alphaF-140sl-1cb-3ft alphaF-140sl-3cb-3ft alphaF-140sl-5cb-3ft alphaF-140sl-6cb-3ft alphaF-160sl-1cb-3ft alphaF-160sl-3cb-3ft alphaF-160sl-5cb-3ft alphaF-160sl-6cb-3ft alphaF-180sl-1cb-3ft alphaF-180sl-3cb-3ft alphaF-180sl-5cb-3ft alphaF-180sl-6cb-3ft alphaF-180sl-7cb-3ft alphaF-200sl-1cb-3ft alphaF-200sl-3cb-3ft alphaF-200sl-5cb-3ft alphaF-200sl-6cb-3ft alphaF-200sl-7cb-3ft" # list of availability matrices
availabilities="alphaF-200sl-2cb-3ft alphaF-200sl-4cb-3ft" # list of availability matrices


# availabilities="alphaF-0cb-3ft alphaF-120sl-1cb-3ft alphaF-120sl-2cb-3ft alphaF-120sl-3cb-3ft alphaF-120sl-4cb-3ft alphaF-120sl-5cb-3ft alphaF-120sl-6cb-3ft alphaF-140sl-1cb-3ft alphaF-140sl-2cb-3ft alphaF-140sl-3cb-3ft alphaF-140sl-4cb-3ft alphaF-140sl-5cb-3ft alphaF-140sl-6cb-3ft alphaF-160sl-1cb-3ft alphaF-160sl-2cb-3ft alphaF-160sl-3cb-3ft alphaF-160sl-4cb-3ft alphaF-160sl-5cb-3ft alphaF-160sl-6cb-3ft alphaF-180sl-1cb-3ft" # list of availability matrices
# availabilities="alphaF-180sl-2cb-3ft alphaF-180sl-3cb-3ft alphaF-180sl-4cb-3ft alphaF-180sl-5cb-3ft alphaF-180sl-6cb-3ft alphaF-180sl-7cb-3ft alphaF-200sl-1cb-3ft alphaF-200sl-2cb-3ft alphaF-200sl-3cb-3ft alphaF-200sl-4cb-3ft alphaF-200sl-5cb-3ft alphaF-200sl-6cb-3ft alphaF-200sl-7cb-3ft" # list of availability matrices
# availabilities="alphaF-120sl-1cb-3ft alphaF-120sl-6cb-3ft" # list of availability matrices

# Does the av. mat. include a fine-tuning phase?
fine_tuning=3 # number of finetuning steps

# How many training rounds does it include?
n_rounds="100" # number of training rounds
###########################

### FL ALGORITHM ###
# Which FL algorithm are you using?
fl_algo="fedavg" # space separated names of FL algorithms
# Is the algorithm unbiased?
biased="2" # 0:unbiased, 1:biased, 2:hybrid (unbiased except when all clients available)
####################

### TRAINING PARAMETERS ###
grad_clip_threshold=None # Change this to None if you don't want to clip
verbose=1 # 0,1,2
seeds="42 78 84"
lrs="5e-2 1e-2" # list of learning rates
# lrs="1e-1" # list of learning rates
###########################

#############################################


################
### TRAINING ###
################
cd ../../fl_training
echo "=> training"

participation="1.0"
heterogeneities="0.0"
weights="0.5" # is the beta parameter in the FedStale paper
device="cuda"

# ------------------------------ #
# --- Experiments for FedAvg --- #
# ------------------------------ #
# known participation probs:
if echo "$fl_algo" | grep -q "fedavg"; then
# if [[ "fedavg" == *"$fl_algo"* ]]; then
for availability in $availabilities; do
availability_matrix_path="../availability_matrices/av-mat_${availability}.csv"
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
cifar10 \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--model_name custom \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer sgd \
--logs_dir ../logs/${logs_folder}/${availability}/biased_${biased}/fedavg/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose ${verbose} \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased} \
--fine_tuning ${fine_tuning} \
# --grad_clip_threshold ${grad_clip_threshold}
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
if echo "$availability" | grep -qE '[0-9]+sl'; then
    n_rounds=$(echo "$availability" | sed -n 's/.*-\([0-9]\+\)sl.*/\1/p')
else
    n_rounds=100
fi
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for seed in $seeds; do
echo "Run FedVARP : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
(
python train.py \
cifar10 \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--model_name custom \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer history \
--logs_dir ../logs/${logs_folder}/${availability}/biased_${biased}/fedvarp/alpha_${alpha}/lr_${lr}/seed_${seed} \
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
if echo "$availability" | grep -qE '[0-9]+sl'; then
    n_rounds=$(echo "$availability" | sed -n 's/.*-\([0-9]\+\)sl.*/\1/p')
else
    n_rounds=100
fi
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
--model_name custom \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer history \
--history_coefficient ${weight} \
--logs_dir ../logs/${logs_folder}/${availability}/biased_${biased}/fedstale/alpha_${alpha}/lr_${lr}/seed_${seed} \
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

