import java.io.*;

public class ModelIO {
    public static void saveModel(DecisionTreeNode root, String filename) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(root);
        }
    }

    public static DecisionTreeNode loadModel(String filename) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            return (DecisionTreeNode) ois.readObject();
        }
    }
}