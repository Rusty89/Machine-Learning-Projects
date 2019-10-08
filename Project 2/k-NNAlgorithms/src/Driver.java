import java.io.File;
import java.util.ArrayList;

public class Driver {

    private static final int numTrainingSets = 10; // number of training sets for 10-fold cross validation

    public static void main(String args[])throws Exception {

        // categorical sets
        Data car = new CarData(new File("./DataSets/car.data")); //use hamming distance (euclidean = false) as all features are categorical
        Data abalone = new AbaloneData(new File("./DataSets/abalone.data")); //use euclidean
        Data segmentation = new ImageData(new File("./DataSets/segmentation.data")); //use euclidean

        // regresssion sets, use regression and euclidean for all these sets
        Data forestFire = new FireData(new File("./DataSets/forestfires.data"));
        Data machine = new MachineData(new File("./DataSets/machine.data"));
        Data redWine = new WineData(new File("./DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("./DataSets/winequality-white.csv"));

        System.out.println("\nBegin Car tests:");
        car.runTests(false,false,4);

        System.out.println("\nBegin Abalone tests:");
        abalone.runTests(false,true,3);

        System.out.println("\nBegin Segmentation tests:");
        segmentation.runTests(false,true,7);

        System.out.println("\nBegin Forest Fire tests:");
        forestFire.runTests(true,true,10);

        System.out.println("\nBegin Machine HW tests:");
        machine.runTests(true,true,10);

        System.out.println("\nBegin Red Wine tests:");
        redWine.runTests(true,true,10);

        System.out.println("\nBegin White Wine tests:");
        whiteWine.runTests(true,true,10);

    }
}
