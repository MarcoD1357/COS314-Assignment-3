# ArithmeticGP — Genetic Programming Arithmetic Classifier
### COS314 Assignment 3 | Breast Cancer Wisconsin (Diagnostic) Dataset

---

## Requirements

- Java JDK 8 or higher
- The following files in the same directory:
  - `ArithmeticGP.java`
  - `Breast_train.csv`
  - `Breast_test.csv`

---

## Step 1: Compile

```bash
javac ArithmeticGP.java
```

---

## Step 2: Train the Model

```bash
java ArithmeticGP train   
```

**Example:**
```bash
java ArithmeticGP train 42 Breast_train.csv best_model.txt
```

**What this does:**
- Runs 100 generations of GP evolution on the training data
- Prints accuracy, F1 score (at threshold 0), and tree depth at every generation
- After evolution, finds the best classification threshold on the training set
- Saves the best evolved model AND its optimal threshold to the output file

**What you'll see per generation:**
Gen   0: Acc = 0.9071 | F1 (t=0) = 0.7234 | Depth = 1 | Expression: X4
Gen   1: Acc = 0.9235 | F1 (t=0) = 0.7891 | Depth = 6 | Expression: ...
...
Gen 100: Acc = 0.9126 | F1 (t=0) = 0.8102 | Depth = 6 | Expression: ...
Optimised threshold: 2.3
Training complete! Runtime: 0.488 seconds
Best model saved to best_model.txt

**Note:** The model file stores the threshold on the first line, followed by the expression. Do not edit it manually.

---

## Step 3: Test the Model

```bash
java ArithmeticGP test <model_input_file> <test_csv>
```

**Example:**
```bash
java ArithmeticGP test best_model.txt Breast_test.csv
```

**What you'll see:**
--- Testing Evolved Model ---
Accuracy:  75.58%
F-measure: 0.7123
Test Runtime: 0.004 seconds

---

## Step 4: Running 30 Independent Experiments (For the Report)

The assignment requires 30 independent runs with unique seeds. A script is provided:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

This will:
- Train and test the model 30 times using seeds 1–30
- Save each model as `model_seed_<seed>.txt`
- Output all results to `results_arithmetic.csv`

After it finishes, open `results_arithmetic.csv` to find the best performing run by Test Accuracy. **That seed is the one to use for your demo.**

---

## Demo Day

Run the single best seed found from your 30 runs. Example if seed 7 was best:

```bash
# Train
java ArithmeticGP train 7 Breast_train.csv best_model.txt

# Test
java ArithmeticGP test best_model.txt Breast_test.csv
```

Record the seed in your report so results can be replicated exactly.

---

## GP Parameters (for the report)

| Parameter                | Value                                      |
|--------------------------|--------------------------------------------|
| Population Size          | 200                                        |
| Max Generations          | 100                                        |
| Initial Tree Depth       | 2–6 (ramped half-and-half)                 |
| Max Tree Depth           | 6                                          |
| Function Set             | +, −, ×, ÷ (protected division)           |
| Terminal Set             | Features X0–X8, random constants [−5, 5]  |
| Selection                | Tournament (size 7)                        |
| Crossover Rate           | 85%                                        |
| Mutation Rate            | 15%                                        |
| Mutation Type            | Point mutation                             |
| Fitness Function         | Accuracy                                   |
| Elitism                  | Yes (best individual preserved each gen)   |
| Classification Threshold | Optimised post-evolution on training set   |

---

## Dataset Notes

The dataset has been pre-encoded as integers (see assignment appendix). The label column is the last column:
- `0` = no-recurrence-events
- `1` = recurrence-events

Missing values (`?`) are treated as `0.0`.

---

## File Structure
.
├── ArithmeticGP.java        # Source code
├── Breast_train.csv         # Training data
├── Breast_test.csv          # Test data
├── run_experiments.sh       # 30-run batch script
├── results_arithmetic.csv   # Generated after running experiments
├── best_model.txt           # Generated after training (threshold + expression)
└── README.md                # This file

---

## Troubleshooting

**`javac: command not found`** — Java is not installed or not on your PATH. Install JDK 8+.

**`Error: Could not find or load main class ArithmeticGP`** — Make sure you compiled first (`javac ArithmeticGP.java`) and are running from the same directory.

**`FileNotFoundException`** — Check that your CSV file paths are correct and the files are in the current directory.

**Model file from old version doesn't load** — The current model file format stores the threshold on the first line. Old model files without a threshold line will fail to load. Just retrain to generate a new model file.