#!/bin/bash

echo "=> generate data"

cd fl_training/data/mnist || exit 1
rm -rf all_data
python generate_data.py \
    --n_tasks 7 \
    --s_frac 1.0 \
    --by_labels_split \
    --alpha 0.5 \
    --seed 12345

cd ../../../fl_training
# The budget should take the values (if the number of fine-tuning steps is 3) 7.980065999999999 6.1960109999999995 4.235079 1.983414 1.773765 0.9428550000000001 0.7447260000000001
# If the number of fine-tuning steps is 1 then it is -> 7.980065999999999 6.1960109999999995 4.235079 1.983414 0.9428550000000001 0.7447260000000001 0.5575350000000001
t_sl_values=(50 90 50 40 40 50 40)
T_values=(40 30 20 10 9 5 4)
for ix in {0..6}; do
echo "3ft and ${ix}"
(python offline_stats_and_greedy_baselines.py \
   --experiment mnist \
   --model-name custom \
   --device cpu \
   --bz 128 \
   --probe-fraction 0.05 \
   --min-probe-samples 32 \
   --seed 12345 \
   --T ${T_values[$ix]} \
   --t-sl ${t_sl_values[$ix]} \
   --t-ft 3 \
   --alpha 0.1 \
   --budget ${ix} \
   --countries "France,Great Britain,Spain,Germany,Belgium,Sweden,Finland" \
   --no-random-start \
   --start-time "2022-01-01T00:00:00" \
   --out-dir "../avMat_baselines_mnist_bestEndFT" \
   --plot \
   --name-prefix "av-mat"
)
done

t_sl_values=(110 70 50 40 50 50 70)
T_values=(40 30 20 10 5 4 3)
for ix in {0..6}; do
echo "3ft and ${ix}"
(python offline_stats_and_greedy_baselines.py \
   --experiment mnist \
   --model-name custom \
   --device cpu \
   --bz 128 \
   --probe-fraction 0.05 \
   --min-probe-samples 32 \
   --seed 12345 \
   --T ${T_values[$ix]} \
   --t-sl ${t_sl_values[$ix]} \
   --t-ft 1 \
   --alpha 0.1 \
   --budget ${ix} \
   --countries "France,Great Britain,Spain,Germany,Belgium,Sweden,Finland" \
   --no-random-start \
   --start-time "2022-01-01T00:00:00" \
   --out-dir "../avMat_baselines_mnist_bestEndFT" \
   --plot \
   --name-prefix "av-mat"
)
done


#    --experiment mnist \
#    --model-name mnist_cnn \
#    --out-dir ../avMat_baselines_mnist_bestEndFT \
