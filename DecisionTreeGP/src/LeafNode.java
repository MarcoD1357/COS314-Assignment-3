import java.util.Random;

public class LeafNode implements DecisionTreeNode {
    private int label;

    public LeafNode(int label) {
        this.label = label;
    }
    public int getLabel() { return label; }
    @Override
    public int classify(double[] features) {
        if (label != 0 && label != 1) {
        System.err.println("ERROR: classify returned " + label   + " - fixing to 0");
        return 0;
    }
        return label;
    }

    @Override
    public DecisionTreeNode copy() {
        return new LeafNode(label);
    }

    @Override
    public int depth() {
        return 1;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public DecisionTreeNode mutatePoint(Random rng, int maxDepth, double[][] trainFeatures, int[] trainLabels) {
        return new LeafNode(rng.nextInt(2));
    }

    @Override
    public String toString(String indent) {
        return indent + "class " + label;
    }

    @Override
    public String toString() {
        return "class_" + label;
    }
}