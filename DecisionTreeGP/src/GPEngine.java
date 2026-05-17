import java.util.*;

public class GPEngine {
    private final GPParameters params;
    private final Random rng;
    private final double[][] trainFeatures;
    private final int[] trainLabels;
    private final double[][] testFeatures;
    private final int[] testLabels;
    private final boolean verbose;

    private List<Individual> population;
    private Individual bestEver;  // Track best individual across all generations

    public GPEngine(long seed, GPParameters params,
                    double[][] trainFeatures, int[] trainLabels,
                    double[][] testFeatures, int[] testLabels,
                    boolean verbose) {
        this.params = params;
        this.rng = new Random(seed);
        this.trainFeatures = trainFeatures;
        this.trainLabels = trainLabels;
        this.testFeatures = testFeatures;
        this.testLabels = testLabels;
        this.verbose = verbose;
        this.bestEver = null;
    }

    public Individual evolve() {
        // Initialize population
        initializePopulation();
        
        // Evaluate initial population
        for (Individual ind : population) {
            evaluateFitness(ind);
        }
        
        // Find best individual in initial population
        bestEver = findBest(population).copy();
        
        if (verbose) {
            double testAcc = evaluateOnTest(bestEver);
            System.out.printf("Gen 0: best train acc = %.4f, test acc = %.4f, size=%d, depth=%d%n",
                    bestEver.getFitness(), testAcc, bestEver.size(), bestEver.depth());
            System.out.println("Best individual:\n" + bestEver);
        }

        // Evolution loop
        for (int gen = 1; gen <= params.maxGenerations; gen++) {
            List<Individual> newPopulation = new ArrayList<>();
            
            // Elitism: keep the best individuals
            List<Individual> sorted = new ArrayList<>(population);
            sorted.sort(Collections.reverseOrder());
            for (int i = 0; i < params.elitismCount && i < sorted.size(); i++) {
                newPopulation.add(sorted.get(i).copy());
            }
            
            // Create offspring
            while (newPopulation.size() < params.populationSize) {
                Individual parent1 = tournamentSelect(population);
                Individual parent2 = tournamentSelect(population);
                
                Individual offspring;
                if (rng.nextDouble() < params.crossoverRate) {
                    offspring = crossover(parent1, parent2);
                } else {
                    offspring = parent1.copy();
                }
                
                if (rng.nextDouble() < params.mutationRate) {
                    offspring = mutate(offspring);
                }
                
                newPopulation.add(offspring);
            }
            
            // Replace population
            population = newPopulation;
            
            // Evaluate new population
            for (Individual ind : population) {
                evaluateFitness(ind);
            }
            
            // Find best in current generation
            Individual currentBest = findBest(population);
            
            // Update global best if current is better
            if (currentBest.getFitness() > bestEver.getFitness()) {
                bestEver = currentBest.copy();
            }
            
            if (verbose) {
                double testAcc = evaluateOnTest(currentBest);
                System.out.printf("Gen %d: best train acc = %.4f, test acc = %.4f, size=%d, depth=%d%n",
                        gen, currentBest.getFitness(), testAcc, currentBest.size(), currentBest.depth());
                System.out.println("Best individual:\n" + currentBest);
            }
            
            // Early stopping if perfect accuracy
            if (bestEver.getFitness() >= 0.9999) {
                break;
            }
        }
    return bestEver.copy();
    }
    
    // Evaluate fitness for a single individual
private void evaluateFitness(Individual ind) {
    int correct = 0;
    
    for (int i = 0; i < trainFeatures.length; i++) {
        int prediction = ind.classify(trainFeatures[i]);
        
        if (prediction == trainLabels[i]) {
            correct++;
        }
    }
    double accuracy = (double) correct / (double) trainFeatures.length;
    ind.setFitness(accuracy);

}


    
    // Find best individual in a population
    private Individual findBest(List<Individual> pop) {
        Individual best = pop.get(0);
        for (Individual ind : pop) {
            if (ind.getFitness() > best.getFitness()) {
                best = ind;
            }
        }
        return best;
    }
    
    // Evaluate on test set
    private double evaluateOnTest(Individual ind) {
        int correct = 0;
        for (int i = 0; i < testFeatures.length; i++) {
            int prediction = ind.classify(testFeatures[i]);
            if (prediction == testLabels[i]) {
                correct++;
            }
        }
        return (double) correct / testFeatures.length;
    }
    
    // Initialize population using ramped half-and-half
    private void initializePopulation() {
        population = new ArrayList<>(params.populationSize);
        
        int minDepth = params.initMinDepth;
        int maxDepth = params.initMaxDepth;
        int depthRange = maxDepth - minDepth + 1;
        
        for (int i = 0; i < params.populationSize; i++) {
            int depth = minDepth + (i % depthRange);
            boolean useFull = (i % 2 == 0); // Half full, half grow
            DecisionTreeNode tree = generateRandomTree(depth, useFull);
            population.add(new Individual(tree));
        }
    }
    
    // Generate random tree
private DecisionTreeNode generateRandomTree(int maxDepth, boolean full) {
    if (maxDepth <= 1) {
        // Use nextDouble to get a true 50/50 split
        return new LeafNode(rng.nextDouble() < 0.5 ? 0 : 1);
    }
    // For grow method, sometimes create leaf even if depth allows more
    if (!full && rng.nextDouble() < 0.15) {  // reduced probability
        return new LeafNode(rng.nextDouble() < 0.5 ? 0 : 1);
    }
    // Create test node
    return TestNode.randomTestNode(rng, maxDepth - 1, trainFeatures, trainLabels);
}
    
    // Tournament selection
    private Individual tournamentSelect(List<Individual> pop) {
        Individual best = null;
        for (int i = 0; i < params.tournamentSize; i++) {
            int idx = rng.nextInt(pop.size());
            Individual contender = pop.get(idx);
            if (best == null || contender.getFitness() > best.getFitness()) {
                best = contender;
            }
        }
        return best.copy();
    }
    
    // Crossover operator
    private Individual crossover(Individual parent1, Individual parent2) {
        DecisionTreeNode childRoot = parent1.getRoot().copy();
        DecisionTreeNode donorRoot = parent2.getRoot().copy();
        
        // Get all nodes from child
        List<DecisionTreeNode> childNodes = new ArrayList<>();
        collectNodes(childRoot, childNodes);
        
        if (childNodes.isEmpty()) {
            return parent1.copy();
        }
        
        // Select random node in child to replace
        int childIdx = rng.nextInt(childNodes.size());
        DecisionTreeNode childNode = childNodes.get(childIdx);
        
        // Get all nodes from donor
        List<DecisionTreeNode> donorNodes = new ArrayList<>();
        collectNodes(donorRoot, donorNodes);
        
        if (donorNodes.isEmpty()) {
            return parent1.copy();
        }
        
        // Select random node from donor
        int donorIdx = rng.nextInt(donorNodes.size());
        DecisionTreeNode donorSubtree = donorNodes.get(donorIdx).copy();
        
        // Find parent of childNode to replace it
        replaceNode(childRoot, childNode, donorSubtree);
        
        return new Individual(childRoot);
    }
    
    // Mutation operator
    private Individual mutate(Individual ind) {
        DecisionTreeNode root = ind.getRoot().copy();
        
        // Get all nodes
        List<DecisionTreeNode> nodes = new ArrayList<>();
        collectNodes(root, nodes);
        
        if (nodes.isEmpty()) {
            return ind.copy();
        }
        
        // Select random node to mutate
        int idx = rng.nextInt(nodes.size());
        DecisionTreeNode nodeToMutate = nodes.get(idx);
        
        // Create mutated node
        DecisionTreeNode mutatedNode;
        if (nodeToMutate instanceof LeafNode) {
            // Flip leaf class
            LeafNode leaf = (LeafNode) nodeToMutate;
            int currentClass = leaf.classify(new double[9]); // Hack to get class
            // Actually, better to store label in LeafNode with a getter
            mutatedNode = new LeafNode(rng.nextInt(2));
        } else {
            // Replace test node with new random test node
            mutatedNode = TestNode.randomTestNode(rng, params.maxDepth, trainFeatures, trainLabels);
        }
        
        // Replace the node
        replaceNode(root, nodeToMutate, mutatedNode);
        
        return new Individual(root);
    }
    
    // Helper: collect all nodes in a tree
    private void collectNodes(DecisionTreeNode node, List<DecisionTreeNode> nodes) {
        if (node == null) return;
        nodes.add(node);
        if (node instanceof TestNode) {
            TestNode tn = (TestNode) node;
            collectNodes(tn.getLeft(), nodes);
            collectNodes(tn.getRight(), nodes);
        }
    }
    
    // Helper: replace a node in the tree
    private void replaceNode(DecisionTreeNode root, DecisionTreeNode target, DecisionTreeNode replacement) {
        if (root == target) {
            // Can't replace root directly through this method
            return;
        }
        replaceNodeHelper(root, target, replacement);
    }
    
    private boolean replaceNodeHelper(DecisionTreeNode current, DecisionTreeNode target, DecisionTreeNode replacement) {
        if (current == null) return false;
        
        if (current instanceof TestNode) {
            TestNode tn = (TestNode) current;
            if (tn.getLeft() == target) {
                tn.setLeft(replacement);
                return true;
            }
            if (tn.getRight() == target) {
                tn.setRight(replacement);
                return true;
            }
            if (replaceNodeHelper(tn.getLeft(), target, replacement)) return true;
            if (replaceNodeHelper(tn.getRight(), target, replacement)) return true;
        }
        return false;
    }
}