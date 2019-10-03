import java.io.File;
import java.util.ArrayList;

public class Driver {

    public static void main(String args[])throws Exception{

        Data machine = new MachineData(new File("../DataSets/machine.data"));
        Data car = new CarData(new File("../DataSets/car.data"));
        Data abalone = new AbaloneData(new File("../DataSets/abalone.data"));
        Data forestFire = new FireData(new File("../DataSets/forestfires.data"));
        Data segmentation = new ImageData(new File("../DataSets/segmentation.data"));
        Data redWine = new WineData(new File("../DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("../DataSets/winequality-white.csv"));

        for (int i = 0; i <10 ; i++) {
            ArrayList<String> result1 = Algorithms.KNN(whiteWine.dataSets.trainingSets.get(i), whiteWine.dataSets.validationSets.get(i), whiteWine.dataSets.testSets.get(i), 3, false,false );
            System.out.println(MathFunction.rootMeanSquaredError(result1, whiteWine.dataSets.testSets.get(i)));
        }

        System.out.println();

    }
}
