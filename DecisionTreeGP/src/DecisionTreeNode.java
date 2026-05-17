import java.io.Serializable;
import java.util.Random;

public interface DecisionTreeNode extends Serializable {
    int classify(double[] features);
    DecisionTreeNode copy();
    int depth();
    int size();
    DecisionTreeNode mutatePoint(Random rng, int maxDepth, double[][] trainFeatures, int[] trainLabels);
    String toString(String indent);
}