#!/bin/bash

GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"

EXEC="python mlp.py"

run_test() {
    NAME=$1
    EXPECT_FAIL=$2
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
echo "                MLP Training Tests"
echo "===================================================="

echo ""
echo "-------------- FILE configuration -----------------"
run_test "train minimal" false $EXEC --dataset datasets/train_set.csv --config config/simple_config.txt
run_test "train with config file subject" false $EXEC --dataset datasets/train_set.csv --config config/subject_config.txt

echo ""
echo "-------------- CLI configuration -----------------"
run_test "train minimal" false $EXEC --dataset datasets/train_set.csv --layer 2 2 --epochs 1 --batch_size 2
run_test "train with config flag subject" false $EXEC --dataset datasets/train_set.csv  --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy --batch_size 8 --learning_rate 0.0314
run_test "train categorical" false $EXEC --dataset datasets/train_set.csv --layer 2 2 --epochs 1 --batch_size 2 --loss categoricalCrossentropy --output_size 2 --activation_output softmax
run_test "train binary" false $EXEC --dataset datasets/train_set.csv --layer 2 2 --epochs 1 --batch_size 2 --loss binaryCrossentropy --output_size 1 --activation_output sigmoid

echo ""
echo "-------------- Mixed configuration -----------------"
run_test "train with config file subject and CLI override" false $EXEC --dataset datasets/train_set.csv --config config/subject_config.txt --epochs 1
run_test "train with CLI and config file flag after" false $EXEC --dataset datasets/train_set.csv --epochs 1 --config config/subject_config.txt