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
echo "             Config Parsing Tests"
echo "===================================================="

# Échecs attendus
run_test "config inexistant" true $EXEC --dataset datasets/train_set.csv --config config/missing.txt
run_test "config epochs < 0" true $EXEC --dataset datasets/train_set.csv --config config/invalid_epochs.txt

# Succès attendus
run_test "config valide simple" false $EXEC --dataset datasets/train_set.csv --config config/simple_config.txt
