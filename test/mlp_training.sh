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
        # echo "---- OUTPUT ----"
        # echo "$OUTPUT"
        # echo "----------------"
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

# Succès attendus
run_test "train minimal" false $EXEC --dataset datasets/train_set.csv --layer 2 2 --epochs 1 --batch_size 2

# Succès attendus avec softmax
run_test "train categorical" false $EXEC --dataset datasets/train_set.csv --layer 2 2 --epochs 1 --batch_size 2 --loss categoricalCrossentropy --output_size 2 --activation_output softmax
