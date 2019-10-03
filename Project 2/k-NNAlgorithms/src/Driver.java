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

        ArrayList<String> result1 = Algorithms.KNN(car.dataSets.trainingSets.get(0), car.dataSets.validationSets.get(0), car.dataSets.testSets.get(0), 1, false,false );
        System.out.println(MathFunction.accuracy(result1,car.dataSets.testSets.get(0)));
        System.out.println();

    }
}
