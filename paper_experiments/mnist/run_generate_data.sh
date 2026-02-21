#!/bin/bash
source ../../greenfl_venv_oldpytorch/bin/activate

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
cd fl_training/data/mnist || exit 1

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