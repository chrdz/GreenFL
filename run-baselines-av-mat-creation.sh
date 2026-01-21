

echo "=> generate data"

cd fl_training/data/cifar10 || exit 1
rm -rf all_data
python generate_data.py \
    --n_tasks 7 \
    --s_frac 0.1 \
    --by_labels_split \
    --alpha 0.5 \
    --seed 12345

cd ../../../fl_training
# The budget should take the values (if the number of fine-tuning steps is 3) 7.980065999999999 6.1960109999999995 4.235079 1.983414 1.773765 0.9428550000000001 0.7447260000000001
# If the number of fine-tuning steps is 1 then it is -> 7.980065999999999 6.1960109999999995 4.235079 1.983414 0.9428550000000001 0.7447260000000001 0.5575350000000001
for ix in {5..5}; do
echo "3ft and ${ix}"
(python offline_stats_and_greedy_baselines.py \
   --experiment cifar10 \
   --model-name custom \
   --device cpu \
   --bz 64 \
   --probe-fraction 0.05 \
   --min-probe-samples 32 \
   --seed 12345 \
   --T 100 \
   --t-sl 20 \
   --t-ft 3 \
   --alpha 0.1 \
   --budget ${ix} \
   --countries "France,Great Britain,Spain,Germany,Belgium,Sweden,Finland" \
   --no-random-start \
   --start-time "2022-01-01T00:00:00" \
   --out-dir "../avMat_baselines_cifar10_bestEndFT" \
   --plot \
   --name-prefix "av-mat"
)
done

# For 1 ft
# for ix in {6..6}; do
# echo "1ft and ${ix}"
# (python offline_stats_and_greedy_baselines.py \
#    --experiment cifar10 \
#    --model-name custom \
#    --device cpu \
#    --bz 64 \
#    --probe-fraction 0.05 \
#    --min-probe-samples 32 \
#    --seed 12345 \
#    --T 50 \
#    --t-sl 70 \
#    --t-ft 1 \
#    --alpha 0.1 \
#    --budget ${ix} \
#    --countries "France,Great Britain,Spain,Germany,Belgium,Sweden,Finland" \
#    --no-random-start \
#    --start-time "2022-01-01T00:00:00" \
#    --plot \
#    --name-prefix "av-mat"
# )
# done


#    --experiment mnist \
#    --model-name mnist_cnn \
#    --out-dir ../avMat_baselines_mnist_bestEndFT \
