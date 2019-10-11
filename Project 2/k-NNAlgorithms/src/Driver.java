import java.io.File;

public class Driver {

    public static void main(String args[])throws Exception {

        // read in our categorical sets
        Data car = new CarData(new File("./DataSets/car.data"));
        Data abalone = new AbaloneData(new File("./DataSets/abalone.data"));
        Data segmentation = new ImageData(new File("./DataSets/segmentation.data"));

        // read in our regression sets, use regression and euclidean parameters for all these sets
        Data forestFire = new FireData(new File("./DataSets/forestfires.data"));
        Data machine = new MachineData(new File("./DataSets/machine.data"));
        Data redWine = new WineData(new File("./DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("./DataSets/winequality-white.csv"));

        // set euclidean parameter to false so that hamming distance is used for the Car dataset
        System.out.println("\nBegin Car tests:");
        System.out.println("--------------------\n");
        car.runTests(false,false, "Car");

        System.out.println("\nBegin Abalone tests:");
        System.out.println("--------------------\n");
        abalone.runTests(false,true,"Abalone");

        System.out.println("\nBegin Segmentation tests:");
        System.out.println("--------------------\n");
        segmentation.runTests(false,true,"Segmentation");

        System.out.println("\nBegin Forest Fire tests:");
        System.out.println("--------------------\n");
        forestFire.runTests(true,true, "ForestFire");

        System.out.println("\nBegin Machine HW tests:");
        System.out.println("--------------------\n");
        machine.runTests(true,true,"MachineData");

        System.out.println("\nBegin Red Wine tests:");
        System.out.println("--------------------\n");
        redWine.runTests(true,true,"RedWine");

        System.out.println("\nBegin White Wine tests:");
        System.out.println("--------------------\n");
        whiteWine.runTests(true,true, "WhiteWine");
    }
}