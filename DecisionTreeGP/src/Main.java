import java.util.*;

public class Main {
    private static final int FEATURE_COUNT = 9;

    public static void main(String[] args) {
        if (args.length < 1) {
            printUsage();
            return;
        }

        String mode = args[0];
        switch (mode) {
            case "train":
                trainMode(args);
                break;
            case "test":
                testMode(args);
                break;
            case "runs30":
                runs30Mode(args);
                break;
            default:
                printUsage();
        }
    }

    private static void printUsage() {
        System.out.println("Usage:");
        System.out.println("  java -jar DecisionTreeGP.jar train --seed <seed> --train <train.csv> --model <out.model> [--display]");
        System.out.println("  java -jar DecisionTreeGP.jar test --model <model> --test <test.csv>");
        System.out.println("  java -jar DecisionTreeGP.jar runs30 --train <train.csv> --test <test.csv> --outdir <dir>");
    }

    private static void trainMode(String[] args) {
        long seed = 12345;
        String trainPath = null;
        String modelPath = "dt_model.model";
        boolean display = false;

        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--seed": seed = Long.parseLong(args[++i]); break;
                case "--train": trainPath = args[++i]; break;
                case "--model": modelPath = args[++i]; break;
                case "--display": display = true; break;
            }
        }

        if (trainPath == null) {
            System.err.println("Missing --train file");
            return;
        }

        try {
            double[][] trainFeat = DataLoader.loadFeatures(trainPath, FEATURE_COUNT);
            int[] trainLab = DataLoader.loadLabels(trainPath);
            double[][] dummyTest = new double[0][];
            int[] dummyLab = new int[0];

            GPParameters params = new GPParameters();
            GPEngine engine = new GPEngine(seed, params, trainFeat, trainLab, dummyTest, dummyLab, display);
            Individual best = engine.evolve();

            System.out.println("Training completed. Best training accuracy: " + best.getFitness());
            ModelIO.saveModel(best.getRoot(), modelPath);
            System.out.println("Model saved to " + modelPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void testMode(String[] args) {
        String modelPath = null;
        String testPath = null;

        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--model": modelPath = args[++i]; break;
                case "--test": testPath = args[++i]; break;
            }
        }

        if (modelPath == null || testPath == null) {
            System.err.println("Missing --model or --test");
            return;
        }

        try {
            DecisionTreeNode model = ModelIO.loadModel(modelPath);
            double[][] testFeat = DataLoader.loadFeatures(testPath, FEATURE_COUNT);
            int[] testLab = DataLoader.loadLabels(testPath);

            int tp = 0, tn = 0, fp = 0, fn = 0;
            for (int i = 0; i < testFeat.length; i++) {
                int pred = model.classify(testFeat[i]);
                int actual = testLab[i];
                if (pred == 1 && actual == 1) tp++;
                else if (pred == 0 && actual == 0) tn++;
                else if (pred == 1 && actual == 0) fp++;
                else if (pred == 0 && actual == 1) fn++;
            }

            double accuracy = (tp + tn) / (double) (tp + tn + fp + fn);
            double precision = tp == 0 ? 0 : (double) tp / (tp + fp);
            double recall = tp == 0 ? 0 : (double) tp / (tp + fn);
            double fmeasure = (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);

            System.out.printf("Test results:\n");
            System.out.printf("Accuracy: %.4f\n", accuracy);
            System.out.printf("F-measure: %.4f\n", fmeasure);
            System.out.printf("Confusion: TP=%d, TN=%d, FP=%d, FN=%d\n", tp, tn, fp, fn);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void runs30Mode(String[] args) {
        String trainPath = null;
        String testPath = null;
        String outDir = ".";

        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--train": trainPath = args[++i]; break;
                case "--test": testPath = args[++i]; break;
                case "--outdir": outDir = args[++i]; break;
            }
        }

        if (trainPath == null || testPath == null) {
            System.err.println("Missing --train or --test");
            return;
        }

        try {
            double[][] trainFeat = DataLoader.loadFeatures(trainPath, FEATURE_COUNT);
            int[] trainLab = DataLoader.loadLabels(trainPath);
            double[][] testFeat = DataLoader.loadFeatures(testPath, FEATURE_COUNT);
            int[] testLab = DataLoader.loadLabels(testPath);
            System.out.println("Data check:");
            System.out.println("Training samples: " + trainFeat.length);
            System.out.println("Class distribution:");
            int class0 = 0, class1 = 0;
            for (int label : trainLab) {
                if (label == 0) class0++;
                else class1++;
            }
            System.out.printf("  Class 0: %d, Class 1: %d%n", class0, class1);

            for (int i = 0; i < 3 && i < trainFeat.length; i++) {
                System.out.printf("Sample %d: ", i);
                for (int j = 0; j < trainFeat[i].length; j++) {
                    System.out.printf("%.0f ", trainFeat[i][j]);
                }
                System.out.printf("-> class %d%n", trainLab[i]);
            }
            GPParameters params = new GPParameters();
            double bestTrainAcc = -1;
            Individual bestEverInd = null;
            int bestRun = -1;

            double[] testAccs = new double[30];
            double[] fmeasures = new double[30];
            long[] runtimes = new long[30];

            for (int run = 0; run < 30; run++) {
                long seed = 1000 + run;
                long start = System.currentTimeMillis();
                GPEngine engine = new GPEngine(seed, params, trainFeat, trainLab, testFeat, testLab, false);
                Individual best = engine.evolve();
                long runtime = System.currentTimeMillis() - start;

                double trainAcc = best.getFitness();
                double testAcc = evaluateTest(best, testFeat, testLab);
                double f1 = computeF1(best, testFeat, testLab);

                testAccs[run] = testAcc;
                fmeasures[run] = f1;
                runtimes[run] = runtime;

                System.out.printf("Run %2d: train=%.4f test=%.4f F1=%.4f time=%d ms%n",
                        run, trainAcc, testAcc, f1, runtime);

                if (trainAcc > bestTrainAcc) {
                    bestTrainAcc = trainAcc;
                    bestEverInd = best;
                    bestRun = run;
                }
            }

            String modelFile = outDir + "/best_dt.model";
            ModelIO.saveModel(bestEverInd.getRoot(), modelFile);
            System.out.printf("\nBest run was %d (train acc %.4f). Model saved to %s%n", bestRun, bestTrainAcc, modelFile);
            double meanTest = Arrays.stream(testAccs).average().orElse(0);
            double stdTest = Math.sqrt(Arrays.stream(testAccs).map(v -> (v - meanTest)*(v - meanTest)).average().orElse(0));
            double meanF1 = Arrays.stream(fmeasures).average().orElse(0);
            double meanTime = Arrays.stream(runtimes).average().orElse(0);

            System.out.printf("\nOver 30 runs:\n");
            System.out.printf("Mean test accuracy: %.4f ± %.4f\n", meanTest, stdTest);
            System.out.printf("Mean F1: %.4f\n", meanF1);
            System.out.printf("Mean runtime: %.0f ms\n", meanTime);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static double evaluateTest(Individual ind, double[][] features, int[] labels) {
        int correct = 0;
        for (int i = 0; i < features.length; i++) {
            if (ind.classify(features[i]) == labels[i]) correct++;
        }
        return correct / (double) features.length;
    }

    private static double computeF1(Individual ind, double[][] features, int[] labels) {
        int tp=0, fp=0, fn=0;
        for (int i = 0; i < features.length; i++) {
            int pred = ind.classify(features[i]);
            int actual = labels[i];
            if (pred == 1 && actual == 1) tp++;
            else if (pred == 1 && actual == 0) fp++;
            else if (pred == 0 && actual == 1) fn++;
        }
        double p = (tp+fp==0) ? 0 : (double)tp/(tp+fp);
        double r = (tp+fn==0) ? 0 : (double)tp/(tp+fn);
        return (p+r==0) ? 0 : 2*p*r/(p+r);
    }
}