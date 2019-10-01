import java.io.File;

public class Driver {

    public static void main(String args[])throws Exception{

        Data machine = new MachineData(new File("../DataSets/machine.data"));
        Data car = new CarData(new File("../DataSets/car.data"));
        Data abalone = new AbaloneData(new File("../DataSets/abalone.data"));
        Data forestFire = new FireData(new File("../DataSets/forestfires.data"));
        Data segmentation = new ImageData(new File("../DataSets/segmentation.data"));
        Data redWine = new WineData(new File("../DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("../DataSets/winequality-white.csv"));

    }
}
