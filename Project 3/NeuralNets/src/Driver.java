/* A class that holds our main method for running our algorithms on the dataset. It uses the methods of the Data class
    to perform the algorithms on all data sets after reading in the data from files in the directory.
 */

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class Driver {

    public static void main(String args[])throws Exception {

        /*
        // read in our categorical sets
        Data car = new CarData(new File("./DataSets/car.data"));
        Data abalone = new AbaloneData(new File("./DataSets/abalone.data"));
        Data segmentation = new ImageData(new File("./DataSets/segmentation.data"));

        // read in our regression sets, use regression and euclidean parameters for all these sets
        Data forestFire = new FireData(new File("./DataSets/forestfires.data"));
        Data machine = new MachineData(new File("./DataSets/machine.data"));
        Data redWine = new WineData(new File("./DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("./DataSets/winequality-white.csv"));

        // condense all data using Condensed KNN, K-Means, and K-PAM before passing to networks
        System.out.println("\nCondense all of our data: ");
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> carCondensedDatasets = condenseData(car, false, false);
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> abaloneCondensedDatasets = condenseData(abalone, false, true);
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> segmentationCondensedDatasets = condenseData(segmentation, false, true);
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> forestFireCondensedDatasets = condenseData(forestFire, true, true);
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> machineCondensedDatasets = condenseData(machine, true, true);
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> redWineCondensedDatasets = condenseData(redWine, true, true);
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> whiteWineCondensedDatasets = condenseData(whiteWine, true, true);
        */

        Network testNet = new Network(new int[] {6, 6, 4});
        System.out.println(testNet.getClassNumber());

        Data maccakek = new CarData(new File("DataSets/car.data"));
        int correctAnswers = 0, totalGuesses = 0;

        for (int i = 0; i < 250; i++) {
            for (ArrayList<ArrayList<String>> nonsensical: maccakek.dataSets.trainingSets)
            {
                for (ArrayList<String> entry: nonsensical)
                {
                    testNet.initializeInputLayer(entry);
                    testNet.feedForward();
                    //System.out.println(testNet.getClassNumber() + " guess  |  actual " + Integer.parseInt(entry.get(entry.size() - 1)));

                    if (testNet.guessedCorrectly())
                        correctAnswers++;
                    totalGuesses++;
                    if (totalGuesses % 1000 == 0)
                        System.out.println(((double) correctAnswers / (double) totalGuesses) * 100 + "% correct");
                    testNet.backprop();
                }
            }
        }
    }


    // method to condense to get the 3 condensed datasets for a given full dataset
    public static ArrayList<ArrayList<ArrayList<ArrayList<String>>>> condenseData(Data dataset, boolean regression, boolean euclidean) throws IOException {

        System.out.println("\nBegin "+ dataset.toString() + " tests:");
        System.out.println("--------------------\n");
        dataset.condenseSets(regression, euclidean, dataset.toString());

        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> condensedSets = new ArrayList<ArrayList<ArrayList<ArrayList<String>>>>();

        // run the condensing algorithms
        ArrayList<ArrayList<ArrayList<String>>> condensedSet = dataset.getCondensedSet();
        ArrayList<ArrayList<ArrayList<String>>> kMeansSet = dataset.getKMeansSet();
        ArrayList<ArrayList<ArrayList<String>>> kPamSet = dataset.getKPAMSet();

        condensedSets.add(condensedSet);
        condensedSets.add(kMeansSet);
        condensedSets.add(kPamSet);

        return condensedSets;
    }

    public static void printDataset(ArrayList<ArrayList<ArrayList<ArrayList<String>>>> dataset) {
        for(int i = 0; i < dataset.size(); i++) {
            System.out.println("\nCondensed set: " + i);
            System.out.println(dataset.get(i));
        }
        System.out.println();
    }
}