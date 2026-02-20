#!/bin/bash

GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"

EXEC="python mlp.py"

run_test() {
    NAME=$1
    EXPECT_FAIL=$2  # "true" si l'échec est attendu
    shift 2
    OUTPUT=$($@ 2>&1)
    EXIT=$?

    if [ $EXIT -eq 0 ] && [ "$EXPECT_FAIL" = "false" ]; then
        echo -e "${GREEN}[OK]${NC} $NAME"
    elif [ $EXIT -ne 0 ] && [ "$EXPECT_FAIL" = "true" ]; then
        echo -e "${GREEN}[OK]${NC} $NAME (failure expected)"
    else
        echo -e "${RED}[FAIL]${NC} $NAME (exit code $EXIT)"
        echo "---- OUTPUT ----"
        echo "$OUTPUT"
        echo "----------------"
    fi
}

echo "===================================================="
echo "             CLI Parsing Tests"
echo "===================================================="

# Échecs attendus
echo ""
echo "-------------- FAIL numeric flag -----------------"

run_test "split : non-numeric values" true $EXEC --dataset data.csv --split a,b
run_test "split : value < 0" true $EXEC --dataset data.csv --split -1,0.0
run_test "split : value > 1" true $EXEC --dataset data.csv --split 0.9,0.1

run_test "train : non-numeric layer" true $EXEC --dataset datasets/train_set.csv --layer a 24 24
run_test "train : layer < 0" true $EXEC --dataset datasets/train_set.csv --layer -1 24 24
run_test "train : epochs < 0" true $EXEC --dataset datasets/train_set.csv --epochs -1
run_test "train : learning_rate < 0" true $EXEC --dataset datasets/train_set.csv --learning_rate -0.01
run_test "train : learning_rate > 1" true $EXEC --dataset datasets/train_set.csv --learning_rate 1.1
run_test "train : batch_size < 0" true $EXEC --dataset datasets/train_set.csv --batch_size -1
run_test "train : input_size < 0" true $EXEC --dataset datasets/train_set.csv --input_size -1
run_test "train : input_size < 1" true $EXEC --dataset datasets/train_set.csv --input_size 0.9
run_test "train : output_size < 0" true $EXEC --dataset datasets/train_set.csv --output_size -1
run_test "train : output_size < 1" true $EXEC --dataset datasets/train_set.csv --output_size 0.9

echo ""
echo "-------------- FAIL wrong associated flag -----------------"
run_test "train : wrong activation_output for categoricalCrossentropy" true $EXEC --dataset datasets/train_set.csv --output_size 2 --activation_output sigmoid
run_test "train : wrong output_size for categoricalCrossentropy" true $EXEC --dataset datasets/train_set.csv --loss categoricalCrossentropy --output_size 3 --activation_output softmax
run_test "train : wrong activation_output for categoricalCrossentropy" true $EXEC --dataset datasets/train_set.csv --output_size 1 --activation_output softmax
run_test "train : wrong output_size for categoricalCrossentropy" true $EXEC --dataset datasets/train_set.csv --loss binaryCrossentropy --output_size 1 --activation_output softmax

echo ""
echo "-------------- FAIL wrong flag -----------------"

run_test "split : missing dataset" true $EXEC --split 0.7,0.15

run_test "train : missing dataset" true $EXEC --layer 24 24 24 --epochs 84
run_test "train : wrong dataset" true $EXEC --dataset datasets/test_set.csv --layer 24 24 24 --epochs 84

run_test "test : missing dataset" true $EXEC --predict saved_model.npy
run_test "test : wrong dataset" true $EXEC --dataset datasets/train_set.csv --predict saved_model.npy

# Succès attendus
echo ""
echo "-------------- SUCCESS -----------------"

run_test "without any config given" false $EXEC --dataset datasets/train_set.csv
run_test "config minimale valide" false $EXEC --dataset datasets/train_set.csv --layer 24 24 24 --epochs 84
run_test "loss categoricalCrossentropy valide" false $EXEC --dataset datasets/train_set.csv --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy --output_size 2 --activation_output softmax
