```markdown
# Decision Tree Genetic Programming for Breast Cancer Recurrence

This project implements a **Genetic Programming (GP)** system that evolves decision trees to classify breast cancer recurrence using the Wisconsin Breast Cancer dataset (provided as `Breast_train.csv` and `Breast_test.csv`). The system performs 30 independent runs, reports performance metrics, and saves the best evolved tree.

## Features
- Evolves binary decision trees where internal nodes test `feature < threshold` and leaves predict class 0 (no recurrence) or 1 (recurrence).
- Standard GP parameters: population size 200, tournament selection (size 7), subtree crossover (0.85), point mutation (0.15), elitism (2), max depth 8, ramped half‑and‑half initialisation (depth 3–6).
- Training accuracy, test accuracy, F1‑measure, and runtime are recorded for each run.
- The best model (by training accuracy) is saved as a serialised object (`best_dt.model`).
- A paired t‑test (or Wilcoxon) can be performed externally to compare with the arithmetic GP.

## Requirements
- Java 8 or higher (no external libraries).

## File Structure
```
DecisionTreeGP/
├── src/
│   ├── DataLoader.java
│   ├── DecisionTreeNode.java
│   ├── LeafNode.java
│   ├── TestNode.java
│   ├── GPParameters.java
│   ├── Individual.java
│   ├── GPEngine.java
│   ├── ModelIO.java
│   └── Main.java
├── Breast_train.csv
├── Breast_test.csv
└── README.md
```

## Compilation
From the project root directory, compile all Java files:

```bash
javac -d out src/*.java
```

This creates a `out/` directory containing the compiled `.class` files.

## Running the Program

### 1. Run 30 Independent Experiments
This trains on `Breast_train.csv`, tests on `Breast_test.csv`, and produces summary statistics.

```bash
java -cp out Main runs30 --train Breast_train.csv --test Breast_test.csv --outdir .
```

**Output:**
- Per‑run: training accuracy, test accuracy, F1, runtime (ms).
- Best run index and its training accuracy.
- The best decision tree structure.
- Summary: mean test accuracy ± standard deviation, mean F1, mean runtime.
- The best model is saved as `best_dt.model` in the current directory (or specified `--outdir`).

### 2. Train a Single Model with Verbose Output
Displays the best individual and its metrics at every generation. Useful for debugging or demonstration.

```bash
java -cp out Main train --seed 12345 --train Breast_train.csv --model mymodel.model --display
```

- `--seed`: random seed (any long integer).
- `--model`: file name for the saved model.
- `--display`: shows generation‑by‑generation progress.

### 3. Test a Saved Model
Load a previously evolved model and evaluate it on the test set.

```bash
java -cp out Main test --model best_dt.model --test Breast_test.csv
```

**Output:** test accuracy, F1‑measure, and confusion matrix (TP, TN, FP, FN).

## Parameters (Configurable in `GPParameters.java`)
| Parameter              | Value     | Description                                      |
|------------------------|-----------|--------------------------------------------------|
| `populationSize`       | 200       | Number of decision trees per generation         |
| `maxGenerations`       | 100       | Stopping generation (early stop at 100% acc)    |
| `crossoverRate`        | 0.85      | Probability of applying subtree crossover       |
| `mutationRate`         | 0.15      | Probability of applying point mutation          |
| `tournamentSize`       | 7         | Number of individuals competing in selection    |
| `elitismCount`         | 2         | Best individuals copied unchanged               |
| `maxDepth`             | 8         | Maximum allowed tree depth (after crossover/mut)|
| `initMinDepth`         | 3         | Minimum initial tree depth (ramped half‑and‑half)|
| `initMaxDepth`         | 6         | Maximum initial tree depth                      |
| `useRampedHalfAndHalf` | true      | Use ramped half‑and‑half initialisation         |

## Data Format
The CSV files must contain a header row. The first column is the class label:
- `0` = no recurrence‑events
- `1` = recurrence‑events

The following nine columns are the features (in order), using the integer encodings described in the assignment appendix:

| Column | Feature       | Encoding notes                          |
|--------|---------------|-----------------------------------------|
| 1      | age           | 0:20–29, 1:30–39, 2:40–49, 3:50–59, 4:60–69, 5:70–79 |
| 2      | menopause     | 0:premeno, 1:ge40, 2:lt40              |
| 3      | tumor_size    | integer mapping of binned range         |
| 4      | inv_nodes     | integer mapping of binned range         |
| 5      | node_caps     | 0:no, 1:yes, 2:?                        |
| 6      | deg_malig     | 1,2,3 (degree of malignancy)            |
| 7      | breast        | 0:left, 1:right                         |
| 8      | breast_quad   | 0:left_low, 1:right_up, 2:left_up, 3:right_low, 4:central, 5:? |
| 9      | irradiat      | 0:no, 1:yes                             |

Example row:
```
0,1,0,6,0,0,3,0,0,0
```
means: class 0, age 30–39, premeno, tumor_size=6, inv_nodes=0, node_caps=no, deg_malig=3, breast=left, breast_quad=left_low, irradiat=no.

## How It Works (Briefly)
1. **Initialisation**: Ramped half‑and‑half creates 200 trees with depths between 3 and 6. Leaf nodes are randomly assigned class 0 or 1. Internal nodes are random tests (`feature < threshold`), where thresholds are chosen from actual values in the training data.
2. **Evaluation**: Each tree classifies all training samples. Fitness = accuracy (proportion correct).
3. **Selection**: Tournament selection (size 7) chooses parents.
4. **Crossover**: Subtree crossover swaps a random subtree between two parents, respecting the maximum depth.
5. **Mutation**: Point mutation replaces a node with a new random node of the same arity (leaf becomes a new leaf, test becomes a new test).
6. **Elitism**: The two best individuals survive unchanged.
7. **Termination**: After 100 generations or if perfect accuracy is reached.
8. **30 Runs**: The program performs 30 runs with seeds 1000–1029, records metrics, and keeps the best tree (by training accuracy) as the final model.

## Statistical Comparison
To compare the Decision Tree GP with the Arithmetic GP (implemented separately):
- Collect the 30 test accuracies from each algorithm.
- Perform a paired t‑test (or Wilcoxon signed‑rank test) using a tool like Excel, R, or a simple Java script. The null hypothesis is that the mean test accuracy of the two algorithms is equal.

## Example Output (30 runs)
```
Run  0: train=0.8525 test=0.5581 F1=0.5250 time=1998 ms
Run  1: train=0.8470 test=0.4884 F1=0.4211 time=1228 ms
...
Best run was 14 (train acc 0.8634). Model saved to ./best_dt.model
Best decision tree structure:
(F0 < 1.500)

Over 30 runs:
Mean test accuracy: 0.5407 ± 0.0716
Mean F1: 0.4221
Mean runtime: 1904 ms
```

## Troubleshooting
- **“Class not found”**: Ensure you compiled with `javac -d out src/*.java` and run with `java -cp out Main ...`.
- **File not found**: Place the CSV files in the same directory as the command is executed, or use full paths.
- **OutOfMemoryError**: Reduce population size or max depth. The current settings are modest and should work on any modern machine.

## Authors

```