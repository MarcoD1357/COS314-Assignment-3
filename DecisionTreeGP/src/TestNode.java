import java.util.*;
import java.util.Random;

public class TestNode implements DecisionTreeNode {
    private int attribute;
    private double threshold;
    private DecisionTreeNode left;
    private DecisionTreeNode right;

    public TestNode(int attribute, double threshold, DecisionTreeNode left, DecisionTreeNode right) {
        this.attribute = attribute;
        this.threshold = threshold;
        this.left = left;
        this.right = right;
    }
    public DecisionTreeNode getLeft() { return left; }
    public DecisionTreeNode getRight() { return right; }
    public void setLeft(DecisionTreeNode node) { this.left = node; }
    public void setRight(DecisionTreeNode node) { this.right = node; }
    @Override
    public int classify(double[] features) {
        int result;
        if (features[attribute] < threshold) {
            result = left.classify(features);
        } else {
            result = right.classify(features);
        }
        if (result != 0 && result != 1) {
            System.err.println("ERROR: classify returned " + result + " - fixing to 0");
            return 0;
        }
        return result;
    }


    @Override
    public DecisionTreeNode copy() {
        return new TestNode(attribute, threshold, left.copy(), right.copy());
    }


    @Override
    public int depth() {
        return 1 + Math.max(left.depth(), right.depth());
    }

    @Override
    public int size() {
        return 1 + left.size() + right.size();
    }

    @Override
    public DecisionTreeNode mutatePoint(Random rng, int maxDepth, double[][] trainFeatures, int[] trainLabels) {

        return randomTestNode(rng, maxDepth - 1, trainFeatures, trainLabels);
    }

    public static TestNode randomTestNode(Random rng, int maxDepth, double[][] trainFeatures, int[] trainLabels) {
        int attr = rng.nextInt(9);
        double thresh = randomThreshold(attr, trainFeatures, rng);
        DecisionTreeNode leftSub = randomSubtree(rng, maxDepth, trainFeatures, trainLabels);
        DecisionTreeNode rightSub = randomSubtree(rng, maxDepth, trainFeatures, trainLabels);
        return new TestNode(attr, thresh, leftSub, rightSub);
    }

public static double randomThreshold(int attr, double[][] features, Random rng) {
    Set<Double> uniqueVals = new HashSet<>();
    for (int i = 0; i < features.length; i++) {
        uniqueVals.add(features[i][attr]);
    }
    List<Double> valList = new ArrayList<>(uniqueVals);
    if (valList.isEmpty()) return 0.0;
    
    if (valList.size() > 1 && rng.nextBoolean()) {
        Collections.sort(valList);
        int idx = rng.nextInt(valList.size() - 1);
        return (valList.get(idx) + valList.get(idx + 1)) / 2.0;
    } else {
        return valList.get(rng.nextInt(valList.size()));
    }
}

private static DecisionTreeNode randomSubtree(Random rng, int maxDepth, double[][] trainFeatures, int[] trainLabels) {
    if (maxDepth <= 1) {
        return new LeafNode(rng.nextDouble() < 0.5 ? 0 : 1);
    }
    
    if (rng.nextDouble() < 0.2) {
        return new LeafNode(rng.nextDouble() < 0.5 ? 0 : 1);
    } else {
        return randomTestNode(rng, maxDepth - 1, trainFeatures, trainLabels);
    }
}

    @Override
    public String toString(String indent) {
        String leftStr = left.toString(indent + "  ");
        String rightStr = right.toString(indent + "  ");
        return indent + String.format("if (F%d < %.3f) then\n%s\n%selse\n%s",
                attribute, threshold, leftStr, indent, rightStr);
    }

    @Override
    public String toString() {
        return String.format("(F%d < %.3f)", attribute, threshold);
    }
}