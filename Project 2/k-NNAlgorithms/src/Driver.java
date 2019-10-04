import java.io.File;
import java.util.ArrayList;

public class Driver {

    private static final int numTrainingSets = 10; // number of training sets for 10-fold cross validation

    public static void main(String args[])throws Exception{

        // categorical sets
        Data car = new CarData(new File("./DataSets/car.data")); //use hamming distance (euclidean = false) as all features are categorical
        Data abalone = new AbaloneData(new File("./DataSets/abalone.data")); //use euclidean
        Data segmentation = new ImageData(new File("./DataSets/segmentation.data")); //use euclidean

        // regresssion sets, use regression and euclidean for all these sets
        Data forestFire = new FireData(new File("./DataSets/forestfires.data"));
        Data machine = new MachineData(new File("./DataSets/machine.data"));
        Data redWine = new WineData(new File("./DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("./DataSets/winequality-white.csv"));

        int kValueSelections[] = {1, 3, 5}; // choose odd values for k to avoid tie-breakers

        // test loop for regression data
        // use odd k-values 1, 3, and 5
        for (int k : kValueSelections) {
            double RMSE  = 0; // root mean squared error
            double absError = 0;
            for (int i = 0; i < numTrainingSets ; i++) {
                ArrayList<String> result1 = Algorithms.KNN(machine.dataSets.trainingSets.get(i), machine.dataSets.testSets.get(i), k, true,true   );
                absError += Double.parseDouble(MathFunction.meanAbsoluteError(result1, machine.dataSets.testSets.get(i), machine.fullSet));
                RMSE += Double.parseDouble(MathFunction.rootMeanSquaredError(result1, machine.dataSets.testSets.get(i), machine.fullSet));
            }

            System.out.println("Mean Absolute error is : " + absError / 10 + "  Root Mean Squared Error : " + RMSE / 10);
        }

        // test loop for classification data
        // use odd k-values 1, 3, and 5
        for (int k : kValueSelections) {
            double precisionAvg = 0;
            double recallAvg = 0;
            double accuracyAvg = 0;

            for (int i = 0; i < numTrainingSets ; i++) {
                ArrayList<String> result1 = Algorithms.EditedKNN(segmentation.dataSets.trainingSets.get(i),segmentation.dataSets.testSets.get(i),segmentation.dataSets.validationSets.get(i), k, false,true  );
                result1 = MathFunction.processConfusionMatrix(result1,segmentation.dataSets.testSets.get(i));
                precisionAvg += Double.parseDouble(result1.get(0));
                recallAvg += Double.parseDouble(result1.get(1));
                accuracyAvg += Double.parseDouble(result1.get(2));

            }
            System.out.println("Precision is: " + precisionAvg / 10 + " Recall is:" + recallAvg / 10 + " Accuracy is: " + accuracyAvg / 10);
        }
    }
}
