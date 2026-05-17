import java.io.*;
import java.util.*;

public class ArithmeticGP {

    static Random rand;
    static final int POP_SIZE = 200;
    static final int MAX_GENS = 100;
    static final int MIN_INIT_DEPTH = 2;
    static final int MAX_INIT_DEPTH = 6;
    static final int MAX_DEPTH = 6;
    static final int TOURNAMENT_SIZE = 7;      // design decision
    static final double CROSSOVER_RATE = 0.85;
    static final double MUTATION_RATE = 0.15;

    static int numFeatures = 0;

    // ---------- Main ----------
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            printHelp();
            return;
        }
        String mode = args[0].toLowerCase();

        if (mode.equals("train")) {
            if (args.length != 4) {
                System.out.println("Usage: java ArithmeticGP train <seed> <train_csv> <model_output_file>");
                return;
            }
            long seed = Long.parseLong(args[1]);
            rand = new Random(seed);
            String trainFile = args[2];
            String modelFile = args[3];

            Dataset trainData = loadData(trainFile);
            numFeatures = trainData.features[0].length;

            System.out.println("--- Starting Genetic Programming Training ---");
            long startTime = System.currentTimeMillis();
            Node bestModel = evolve(trainData);
            long endTime = System.currentTimeMillis();

            // ---- Threshold optimisation on training set ----
            double bestThreshold = findBestThreshold(bestModel, trainData);
            System.out.println("Optimised threshold: " + bestThreshold);
            System.out.println("Training complete! Runtime: " + (endTime - startTime) / 1000.0 + " seconds");
            saveModel(bestModel, bestThreshold, modelFile);
            System.out.println("Best model saved to " + modelFile);

        } else if (mode.equals("test")) {
            if (args.length != 3) {
                System.out.println("Usage: java ArithmeticGP test <model_input_file> <test_csv>");
                return;
            }
            String modelFile = args[1];
            String testFile = args[2];

            Object[] loaded = loadModel(modelFile);
            Node model = (Node) loaded[0];
            double threshold = (Double) loaded[1];

            Dataset testData = loadData(testFile);

            System.out.println("--- Testing Evolved Model ---");
            long startTime = System.currentTimeMillis();
            evaluateAndPrintMetrics(model, testData, threshold);
            long endTime = System.currentTimeMillis();
            System.out.println("Test Runtime: " + (endTime - startTime) / 1000.0 + " seconds");
        } else {
            printHelp();
        }
    }

    private static void printHelp() {
        System.out.println("Invalid arguments.");
        System.out.println("TRAIN: java ArithmeticGP train <seed> <train_csv> <model_output_file>");
        System.out.println("TEST:  java ArithmeticGP test <model_input_file> <test_csv>");
    }

    // ---------- Evolution ----------
    public static Node evolve(Dataset data) {
        List<Node> population = initializePopulation();
        Node overallBest = null;
        double overallBestAcc = -1.0;

        for (int gen = 0; gen <= MAX_GENS; gen++) {
            Node genBest = null;
            double genBestAcc = -1.0;

            double[] fitnesses = new double[POP_SIZE];
            for (int i = 0; i < POP_SIZE; i++) {
                fitnesses[i] = calculateAccuracy(population.get(i), data);
                if (fitnesses[i] > genBestAcc) {
                    genBestAcc = fitnesses[i];
                    genBest = population.get(i);
                }
            }

            if (genBestAcc > overallBestAcc) {
                overallBestAcc = genBestAcc;
                overallBest = genBest.clone();
            }

            // Show F1 of the gen-best for monitoring (using default threshold 0)
            double genF1 = calculateF1(genBest, data, 0.0);
            System.out.printf("Gen %3d: Acc = %.4f | F1 (t=0) = %.4f | Depth = %d | Expression: %s%n",
                    gen, genBestAcc, genF1, genBest.getDepth(), genBest.toPrefix());

            if (gen == MAX_GENS) break;

            List<Node> newPopulation = new ArrayList<>();
            newPopulation.add(overallBest.clone());   // elitism

            while (newPopulation.size() < POP_SIZE) {
                if (rand.nextDouble() < CROSSOVER_RATE) {
                    Node p1 = tournamentSelection(population, fitnesses);
                    Node p2 = tournamentSelection(population, fitnesses);
                    Node offspring = crossover(p1, p2);
                    while (offspring.getDepth() > MAX_DEPTH) {
                        offspring = crossover(
                            tournamentSelection(population, fitnesses),
                            tournamentSelection(population, fitnesses)
                        );
                    }
                    newPopulation.add(offspring);
                } else if (rand.nextDouble() < (CROSSOVER_RATE + MUTATION_RATE)) {
                    Node p1 = tournamentSelection(population, fitnesses);
                    Node offspring = pointMutation(p1.clone());   // only point mutation, as per spec
                    newPopulation.add(offspring);
                } else {
                    Node p1 = tournamentSelection(population, fitnesses);
                    newPopulation.add(p1.clone()); // Reproduction
                }
            }
            population = newPopulation;
        }
        return overallBest;
    }

    // ---------- Population initialisation ----------
    public static List<Node> initializePopulation() {
        List<Node> pop = new ArrayList<>();
        int depths = MAX_INIT_DEPTH - MIN_INIT_DEPTH + 1;
        int perDepth = POP_SIZE / depths;

        for (int d = MIN_INIT_DEPTH; d <= MAX_INIT_DEPTH; d++) {
            for (int i = 0; i < perDepth; i++) {
                if (i < perDepth / 2) pop.add(generateTree(d, true));  // Full
                else pop.add(generateTree(d, false));                 // Grow
            }
        }
        while (pop.size() < POP_SIZE) pop.add(generateTree(MAX_INIT_DEPTH, false));
        return pop;
    }

    public static Node generateTree(int depth, boolean full) {
        if (depth == 1 || (!full && rand.nextDouble() < 0.2)) {
            return rand.nextBoolean() ? new VarNode(rand.nextInt(numFeatures))
                                      : new ConstNode(rand.nextDouble() * 10 - 5);
        }
        int op = rand.nextInt(4);
        Node left = generateTree(depth - 1, full);
        Node right = generateTree(depth - 1, full);
        return new FuncNode(op, left, right);
    }

    // ---------- Selection ----------
    public static Node tournamentSelection(List<Node> pop, double[] fitnesses) {
        int bestIdx = rand.nextInt(POP_SIZE);
        for (int i = 1; i < TOURNAMENT_SIZE; i++) {
            int idx = rand.nextInt(POP_SIZE);
            if (fitnesses[idx] > fitnesses[bestIdx]) {
                bestIdx = idx;
            }
        }
        return pop.get(bestIdx);
    }

    // ---------- Crossover ----------
    public static Node crossover(Node p1, Node p2) {
        Node c1 = p1.clone();
        List<Node> nodes1 = c1.getAllNodes();
        List<Node> nodes2 = p2.getAllNodes();

        Node t1 = nodes1.get(rand.nextInt(nodes1.size()));
        Node t2 = nodes2.get(rand.nextInt(nodes2.size())).clone();

        if (t1 == c1) return t2;

        for (Node n : nodes1) {
            if (n instanceof FuncNode) {
                FuncNode fn = (FuncNode) n;
                if (fn.left == t1) { fn.left = t2; break; }
                if (fn.right == t1) { fn.right = t2; break; }
            }
        }
        return c1;
    }

    // ---------- Point mutation only ----------
    public static Node pointMutation(Node node) {
        if (node instanceof FuncNode) {
            if (rand.nextDouble() < 0.1) ((FuncNode) node).op = rand.nextInt(4);
            ((FuncNode) node).left = pointMutation(((FuncNode) node).left);
            ((FuncNode) node).right = pointMutation(((FuncNode) node).right);
        } else if (node instanceof VarNode || node instanceof ConstNode) {
            if (rand.nextDouble() < 0.1) {
                if (rand.nextBoolean()) return new VarNode(rand.nextInt(numFeatures));
                else return new ConstNode(rand.nextDouble() * 10 - 5);
            }
        }
        return node;
    }

    // ---------- Fitness: accuracy (spec requirement) ----------
    public static double calculateAccuracy(Node tree, Dataset data) {
        int correct = 0;
        for (int i = 0; i < data.features.length; i++) {
            double val = tree.eval(data.features[i]);
            int pred = val > 0 ? 1 : 0;
            if (pred == data.labels[i]) correct++;
        }
        return (double) correct / data.labels.length;
    }

    // ---------- Threshold optimisation ----------
    public static double findBestThreshold(Node tree, Dataset data) {
        // Collect all tree outputs
        double[] outputs = new double[data.features.length];
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = tree.eval(data.features[i]);
        }

        // Try a range of candidate thresholds
        double bestThresh = 0.0;
        double bestF1 = -1.0;
        for (double t = -10.0; t <= 10.0; t += 0.1) {
            double f1 = calculateF1(tree, data, t);
            if (f1 > bestF1) {
                bestF1 = f1;
                bestThresh = t;
            }
        }
        return bestThresh;
    }

    // ---------- F1 calculation for a given threshold ----------
    public static double calculateF1(Node tree, Dataset data, double threshold) {
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < data.features.length; i++) {
            double val = tree.eval(data.features[i]);
            int pred = val > threshold ? 1 : 0;
            int actual = data.labels[i];

            if (pred == 1 && actual == 1) tp++;
            else if (pred == 1 && actual == 0) fp++;
            else if (pred == 0 && actual == 1) fn++;
        }
        double precision = (tp + fp) == 0 ? 0 : (double) tp / (tp + fp);
        double recall    = (tp + fn) == 0 ? 0 : (double) tp / (tp + fn);
        return (precision + recall) == 0 ? 0 : 2.0 * precision * recall / (precision + recall);
    }

    // ---------- Evaluation for test (using saved threshold) ----------
    public static void evaluateAndPrintMetrics(Node tree, Dataset data, double threshold) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < data.features.length; i++) {
            double val = tree.eval(data.features[i]);
            int pred = val > threshold ? 1 : 0;
            int actual = data.labels[i];
            if (pred == 1 && actual == 1) tp++;
            else if (pred == 0 && actual == 0) tn++;
            else if (pred == 1 && actual == 0) fp++;
            else if (pred == 0 && actual == 1) fn++;
        }
        double accuracy = (double) (tp + tn) / data.labels.length;
        double precision = (tp + fp) == 0 ? 0 : (double) tp / (tp + fp);
        double recall = (tp + fn) == 0 ? 0 : (double) tp / (tp + fn);
        double f1 = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);

        System.out.printf("Accuracy:  %.2f%%%n", accuracy * 100);
        System.out.printf("F-measure: %.4f%n", f1);
    }

    // ---------- Data loading ----------
    static class Dataset { double[][] features; int[] labels; }

    public static Dataset loadData(String filename) throws Exception {
        List<double[]> featList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;
        boolean headerSkipped = false;
        while ((line = br.readLine()) != null) {
            if (!headerSkipped) { headerSkipped = true; continue; }
            String[] parts = line.split(",");
            if (parts.length < 2) continue;
            double[] f = new double[parts.length - 1];
            for (int i = 1; i < parts.length; i++) {
                String val = parts[i].trim();
                if (val.equals("?")) f[i - 1] = 0.0;
                else try { f[i - 1] = Double.parseDouble(val); } catch (Exception e) { f[i - 1] = 0.0; }
            }
            featList.add(f);
            String labelStr = parts[0].trim().toLowerCase();
            if (labelStr.equals("0")) labelList.add(0);
            else if (labelStr.equals("1")) labelList.add(1);
            else labelList.add(0);
        }
        br.close();
        Dataset d = new Dataset();
        d.features = featList.toArray(new double[0][]);
        d.labels = labelList.stream().mapToInt(i -> i).toArray();
        return d;
    }

    // ---------- Node classes ----------
    static abstract class Node {
        abstract double eval(double[] features);
        public abstract Node clone();
        abstract int getDepth();
        abstract String toPrefix();
        List<Node> getAllNodes() {
            List<Node> list = new ArrayList<>();
            list.add(this);
            if (this instanceof FuncNode) {
                list.addAll(((FuncNode) this).left.getAllNodes());
                list.addAll(((FuncNode) this).right.getAllNodes());
            }
            return list;
        }
    }

    static class FuncNode extends Node {
        int op; // 0:+, 1:-, 2:*, 3:/
        Node left, right;
        FuncNode(int op, Node l, Node r) { this.op = op; left = l; right = r; }
        double eval(double[] f) {
            double l = left.eval(f), r = right.eval(f);
            switch (op) {
                case 0: return l + r;
                case 1: return l - r;
                case 2: return l * r;
                case 3: return r == 0 ? 1 : l / r;
            }
            return 0;
        }
        public Node clone() { return new FuncNode(op, left.clone(), right.clone()); }
        int getDepth() { return 1 + Math.max(left.getDepth(), right.getDepth()); }
        String toPrefix() {
            String o = op == 0 ? "+" : op == 1 ? "-" : op == 2 ? "*" : "/";
            return o + " " + left.toPrefix() + " " + right.toPrefix();
        }
    }

    static class VarNode extends Node {
        int idx;
        VarNode(int idx) { this.idx = idx; }
        double eval(double[] f) { return f[idx]; }
        public Node clone() { return new VarNode(idx); }
        int getDepth() { return 1; }
        String toPrefix() { return "X" + idx; }
    }

    static class ConstNode extends Node {
        double val;
        ConstNode(double val) { this.val = val; }
        double eval(double[] f) { return val; }
        public Node clone() { return new ConstNode(val); }
        int getDepth() { return 1; }
        String toPrefix() { return String.valueOf(val); }
    }

    // ---------- Model persistence (now includes threshold) ----------
    public static void saveModel(Node root, double threshold, String filename) throws Exception {
        BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
        bw.write(threshold + "\n");
        bw.write(root.toPrefix());
        bw.close();
    }

    public static Object[] loadModel(String filename) throws Exception {
        Scanner sc = new Scanner(new File(filename));
        double threshold = Double.parseDouble(sc.nextLine());
        Node root = parsePrefix(sc);
        sc.close();
        return new Object[] { root, threshold };
    }

    private static Node parsePrefix(Scanner sc) {
        if (!sc.hasNext()) return null;
        String token = sc.next();
        if (token.equals("+")) return new FuncNode(0, parsePrefix(sc), parsePrefix(sc));
        if (token.equals("-")) return new FuncNode(1, parsePrefix(sc), parsePrefix(sc));
        if (token.equals("*")) return new FuncNode(2, parsePrefix(sc), parsePrefix(sc));
        if (token.equals("/")) return new FuncNode(3, parsePrefix(sc), parsePrefix(sc));
        if (token.startsWith("X")) return new VarNode(Integer.parseInt(token.substring(1)));
        return new ConstNode(Double.parseDouble(token));
    }
}