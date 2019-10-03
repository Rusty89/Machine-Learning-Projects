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

        //test loop
        for (int i = 0; i <10 ; i++) {
            ArrayList<String> result1 = Algorithms.KNN(forestFire.dataSets.trainingSets.get(i),forestFire.dataSets.testSets.get(i), 3, true,true   );
            System.out.println(MathFunction.meanAbsoluteError(result1, forestFire.dataSets.testSets.get(i)));
        }

        System.out.println();

    }
}
