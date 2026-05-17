import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataLoader {
    public static double[][] loadFeatures(String filePath, int featureCount) throws IOException {
        List<double[]> features = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine();
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double[] feat = new double[featureCount];
                for (int i = 0; i < featureCount; i++) {
                    feat[i] = Double.parseDouble(parts[i + 1]);
                }
                features.add(feat);
            }
        }
        return features.toArray(new double[0][]);
    }

    public static int[] loadLabels(String filePath) throws IOException {
        List<Integer> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine();
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                labels.add(Integer.parseInt(parts[0]));
            }
        }
        return labels.stream().mapToInt(i -> i).toArray();
    }
}