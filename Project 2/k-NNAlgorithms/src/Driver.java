import java.io.File;
import java.util.ArrayList;

public class Driver {

    public static void main(String args[])throws Exception{

        //categorical sets
        Data car = new CarData(new File("../DataSets/car.data"));//use hamming distance (euclidean= false) as all features are categorical
        Data abalone = new AbaloneData(new File("../DataSets/abalone.data"));//use euclidean
        Data segmentation = new ImageData(new File("../DataSets/segmentation.data"));//use euclidean

        //regresssion sets, use regression and euclidean for all these sets
        Data forestFire = new FireData(new File("../DataSets/forestfires.data"));
        Data machine = new MachineData(new File("../DataSets/machine.data"));
        Data redWine = new WineData(new File("../DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("../DataSets/winequality-white.csv"));

        //test loop for regression data
/*
        for (int k = 1; k <20 ; k++) {
            double RMSE=0;
            double absError=0;
            for (int i = 0; i <10 ; i++) {
                ArrayList<String> result1 = Algorithms.KNN(redWine.dataSets.trainingSets.get(i), redWine.dataSets.testSets.get(i), k, true,true   );
                absError+= Double.parseDouble(MathFunction.meanAbsoluteError(result1, redWine.dataSets.testSets.get(i), redWine.fullSet));
                RMSE+= Double.parseDouble(MathFunction.rootMeanSquaredError(result1, redWine.dataSets.testSets.get(i), redWine.fullSet));
            }
            System.out.println(absError/10);
            System.out.println(RMSE/10);
            System.out.println("End set k value ="+k);
        }
*/
        //test loop for classification data
        for (int k = 1; k <40 ; k+=2) {
            double precisionAvg=0;
            double recallAvg=0;
            double accuracyAvg=0;

            for (int i = 0; i <10 ; i++) {
                ArrayList<String> result1 = Algorithms.KNN(segmentation.dataSets.trainingSets.get(i), segmentation.dataSets.testSets.get(i), k, false,true   );
                result1=MathFunction.processConfusionMatrix(result1, segmentation.dataSets.testSets.get(i));
                precisionAvg+=Double.parseDouble(result1.get(0));
                recallAvg+=Double.parseDouble(result1.get(1));
                accuracyAvg+=Double.parseDouble(result1.get(2));

            }
            System.out.println("Precision is: "+ precisionAvg/10+" Recall is:"+recallAvg/10+" Accuracy is: "+accuracyAvg/10);


        }

    }
}
