echo "===================================================="
echo "                  epochs < 0\n"

python mlp.py --dataset datasets/train_set.csv --epochs -1

echo "===================================================="
echo "                  OK\n"

python mlp.py --dataset datasets/train_set.csv --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy --batch_size 8 --learning_rate 0.0314