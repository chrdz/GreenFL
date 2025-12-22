

echo "=> generate data"

cd fl_training/data/mnist || exit 1
rm -rf all_data
python generate_data.py \
    --n_tasks 7 \
    --s_frac 0.1 \
    --by_labels_split \
    --alpha 0.5 \
    --seed 12345

cd ../../..

python fl_training/offline_stats_and_greedy_baselines.py \
   --experiment mnist \
   --model-name mnist_cnn \
   --device cpu \
   --bz 64 \
   --probe-fraction 0.05 \
   --min-probe-samples 32 \
   --seed 12345 \
   --T 100 \
   --t-sl 150 \
   --t-ft 3 \
   --alpha 0.1 \
   --budget 3.0 \
   --countries "France,Great Britain,Spain,Germany,Belgium,Sweden,Finland" \
   --no-random-start \
   --start-time "2022-01-01T00:00:00" \
   --out-dir availability_matrices \
   --plot

