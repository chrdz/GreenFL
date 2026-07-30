#!/bin/bash
#module load conda/2021.11-python3.9
source ../../venv/bin/activate

logs_folder="mnist_alphaFVar_beta0.5_1ft"
av_mat_folder="availability_matrices_alphaFvar_ft"
#######################################################
### - Parameters to choose for dataset generation - ###
#######################################################
# Choose alpha between 0 and 1 to determine the level of non-iid ness of the clients datasets
# This is not the alpha-fairness parameter
alpha="0.5" # distribution of data among clients: 0.1:non-iid, 100000:iid, 0: true iid
generate_data=true #true/false true will regenerate the clients' datasets
# dataseed=12345
# dataseed_folder="dataseed_12345"
# dataseed=12346
# dataseed_folder="dataseed_12346"
dataseed=12347
dataseed_folder="dataseed_12347"
#######################################################


###########################
### DATASETS GENERATION ###
###########################
n_tasks="7" # 7 clients, one client per country

if $generate_data; then
echo "=> generate data"
cd ../..
cd fl_training/data/mnist || exit 1

# --- dataset creation --- #
rm -rf all_data
(
python generate_data.py \
--n_tasks ${n_tasks} \
--s_frac 1.0 \
--test_tasks_frac 0.0 \
--seed ${dataseed} \
--by_labels_split \
--alpha ${alpha}
) # /!\ the two last lines are for non-iid
# --------------------------- #
cd ..
fi


#############################################
### - Parameters to choose for training - ###
#############################################

### AVAILABILITY MATRIX ###
# Which availability matrix/matrices are you using?
# availabilities="alphaF50-alpha0.1-7cb alphaF50-alpha1.0-7cb alphaF50-alpha0.001-8cb alphaF50-alpha0.01-8cb alphaF50-alpha0.1-8cb alphaF50-alpha1.0-8cb alphaF50-alpha0.001-10cb alphaF50-alpha0.01-10cb alphaF50-alpha0.1-10cb alphaF50-alpha1.0-10cb" #list of availability matrices
# availabilities="alphaF50-alpha0.001-7cb" 
availabilities="alphaF50-alpha0.1-10cb-1ft alphaF50-alpha0.01-10cb-1ft alphaF50-alpha0.001-10cb-1ft alphaF50-alpha0.5-10cb-1ft alphaF50-alpha0.9-10cb-1ft alphaF50-alpha0.75-10cb-1ft alphaF50-alpha1.0-10cb-1ft"

# Does the av. mat. include a fine-tuning phase?
# fine_tuning=3 # Change this to # of finetuning step

# How many training rounds does it include?
n_rounds="200" # number of training rounds
###########################

### FL ALGORITHM ###
# Which FL algorithm are you using?
fl_algo="fedavg" # space separated names of FL algorithms
# Is the algorithm unbiased?
biased="2" # 0:unbiased, 1:biased, 2:hybrid (unbiased except when all clients available)
####################

### TRAINING PARAMETERS ###
grad_clip_threshold="1.0" # Change this to None if you don't want to clip
verbose=1 # 0,1,2
seeds="42 78 84"
lrs="1e-2 5e-2" # list of learning rates
###########################

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
availability_matrix_path="../${av_mat_folder}/av-mat_${availability}.csv"
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
--logs_dir ../logs/${logs_folder}/${dataseed_folder}/${availability}/biased_${biased}/fedavg/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose ${verbose} \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased} \
--grad_clip_threshold ${grad_clip_threshold}
)
done
done
done
done
fi