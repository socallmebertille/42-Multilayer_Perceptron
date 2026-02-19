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
run_test "epochs < 0" true $EXEC --dataset datasets/train_set.csv --epochs -1
run_test "batch_size < 0" true $EXEC --dataset datasets/train_set.csv --batch_size -5
run_test "missing dataset" true $EXEC --layer 24 24 24 --epochs 84

# Succès attendus
run_test "without any config given" false $EXEC --dataset datasets/train_set.csv
run_test "config minimale valide" false $EXEC --dataset datasets/train_set.csv --layer 24 24 24 --epochs 84
run_test "loss categoricalCrossentropy valide" false $EXEC --dataset datasets/train_set.csv --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy --output_size 2 --activation_output softmax
