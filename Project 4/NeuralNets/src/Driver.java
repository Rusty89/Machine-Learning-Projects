/* A class that holds our main method for running our algorithms on the dataset. It uses the methods of the Data class
    to perform the algorithms on all data sets after reading in the data from files in the directory.
 */

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

public class Driver {

    private static final int numCondensedSets = 3;
    private static final int numCVSSets = 1;

    public static void main(String[] args) throws Exception {

        // read in our categorical sets
        Data car = new CarData(new File("./DataSets/car.data"));
        Data abalone = new AbaloneData(new File("./DataSets/abalone.data"));
        Data segmentation = new ImageData(new File("./DataSets/segmentation.data"));

        // read in our regression sets, use regression and euclidean parameters for all these sets
        Data forestFire = new FireData(new File("./DataSets/forestfires.data"));
        Data machine = new MachineData(new File("./DataSets/machine.data"));
        Data redWine = new WineData(new File("./DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("./DataSets/winequality-white.csv"));

        GeneticAlgorithm eugene = new GeneticAlgorithm(car.dataSets.trainingSets.get(0), new int[] {6, 3}, 4, false);
        eugene.runAlgorithm();
    }

    private static void printRangeAndMean (String name, ArrayList<Double> results)
    {
        DecimalFormat decimalFormat = new DecimalFormat("#.#########");
        double sum = 0, min = Double.MAX_VALUE, max = Double.MIN_VALUE;
        for (double result: results)
        {
            sum += result;
            if (result < min)
                min = result;
            if (result > max)
                max = result;
        }
        System.out.println(name + " from " + decimalFormat.format(min) + " to " + decimalFormat.format(max) +
                " with a mean of " + decimalFormat.format(sum/results.size()) + ".");
    }
}