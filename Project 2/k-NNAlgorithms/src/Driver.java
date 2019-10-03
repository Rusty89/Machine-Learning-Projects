import java.io.File;
import java.util.ArrayList;

public class Driver {

    public static void main(String args[])throws Exception{

        //categorical sets
        Data car = new CarData(new File("../DataSets/car.data"));//use hamming distance (euclidean= false) as all features are categorical
        Data abalone = new AbaloneData(new File("../DataSets/abalone.data"));//use euclidean
        Data segmentation = new ImageData(new File("../DataSets/segmentation.data"));//use euclidean

        //regresssion sets, use regression and euclidean for all sets
        Data forestFire = new FireData(new File("../DataSets/forestfires.data"));
        Data machine = new MachineData(new File("../DataSets/machine.data"));
        Data redWine = new WineData(new File("../DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("../DataSets/winequality-white.csv"));

        for (int i = 0; i <10 ; i++) {
            ArrayList<String> result1 = Algorithms.KNN(segmentation.dataSets.trainingSets.get(i),segmentation.dataSets.testSets.get(i), 3, false,true   );
            System.out.println(MathFunction.accuracy(result1, segmentation.dataSets.testSets.get(i)));
        }

        System.out.println();

    }
}
