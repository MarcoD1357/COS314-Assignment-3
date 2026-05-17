#!/bin/bash
echo "Seed,Train_Acc,Test_Acc,F_Measure" > results_arithmetic.csv

for seed in {1..30}
do
    # Capture training output and extract Gen 100 accuracy
    train_output=$(java ArithmeticGP train $seed Breast_train.csv model_seed_$seed.txt)
    train_acc=$(echo "$train_output" | grep "^Gen 100" | grep -oP 'Acc = \K[0-9.]+')

    # Run testing
    test_output=$(java ArithmeticGP test model_seed_$seed.txt Breast_test.csv)
    test_acc=$(echo "$test_output" | grep "Accuracy" | grep -oP '[0-9.]+')
    f1=$(echo "$test_output" | grep "F-measure" | grep -oP '[0-9.]+')

    echo "$seed,$train_acc,$test_acc,$f1" >> results_arithmetic.csv
    echo "Seed $seed done: Train=$train_acc Test=$test_acc F1=$f1"
done