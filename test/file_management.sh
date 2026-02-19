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
echo "        Dataset / File Management Tests"
echo "===================================================="

# Échecs attendus
run_test "dataset inexistant" true $EXEC --dataset datasets/missing.csv --layer 24 24 24 --epochs 10
run_test "train_set.csv mais mode predict" true $EXEC --dataset datasets/train_set.csv --predict saved_model.npy
run_test "test_set.csv sans predict" true $EXEC --dataset datasets/test_set.csv

# Succès attendus
run_test "dataset valide train_set.csv" false $EXEC --dataset datasets/train_set.csv --layer 24 24 24 --epochs 1
run_test "dataset valide test_set.csv mode predict" false $EXEC --dataset datasets/test_set.csv --predict saved_model.npy
