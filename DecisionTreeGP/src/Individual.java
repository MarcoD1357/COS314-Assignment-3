public class Individual implements Comparable<Individual> {
    private DecisionTreeNode root;
    private double fitness;

    public Individual(DecisionTreeNode root) {
        this.root = root;
    }

    public DecisionTreeNode getRoot() { return root; }
    public double getFitness() { return fitness; }
    public void setFitness(double fitness) { this.fitness = fitness; }

    public int classify(double[] features) {
        return root.classify(features);
    }

    public Individual copy() {
    Individual copy = new Individual(root.copy());
    copy.setFitness(this.fitness);
    return copy;
}

    public int depth() { return root.depth(); }
    public int size() { return root.size(); }

    @Override
    public int compareTo(Individual other) {
        return Double.compare(other.fitness, this.fitness);
    }

    @Override
    public String toString() {
        return root.toString();
    }
}