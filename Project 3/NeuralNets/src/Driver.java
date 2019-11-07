/* A class that holds our main method for running our algorithms on the dataset. It uses the methods of the Data class
    to perform the algorithms on all data sets after reading in the data from files in the directory.
 */

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Scanner;

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

        // read in our categorical sets
        Data car = new CarData(new File("DataSets/car.data"));
        Data abalone = new AbaloneData(new File("DataSets/abalone.data"));
        Data segmentation = new ImageData(new File("DataSets/segmentation.data"));

        // read in our regression sets, use regression and euclidean parameters for all these sets
        Data forestFire = new FireData(new File("DataSets/forestfires.data"));
        Data machine = new MachineData(new File("DataSets/machine.data"));
        Data redWine = new WineData(new File("DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("DataSets/winequality-white.csv"));

        ArrayList<Data> datasets = new ArrayList<>();

        // We can comment out which datasets we want to run here for video
        datasets.add(car);
        datasets.add(abalone);
        datasets.add(segmentation);
        datasets.add(forestFire);
        datasets.add(machine);
        datasets.add(redWine);
        datasets.add(whiteWine);

        for (Data set: datasets) {
            // User input for setting up network
            System.out.print("How many hidden layers? ");
            Scanner in = new Scanner(System.in);
            String input = in.nextLine();
            int hl = Integer.parseInt(input);
            int[] hlayers = new int[hl + 2];
            for (int i = 0; i < hl; i++) {
                System.out.print("How many nodes in hidden layer " + (i + 1) + "? ");
                input = in.nextLine();
                hlayers[i + 1] = Integer.parseInt(input);
            }

            int inputSize = set.dataSets.trainingSets.get(0).get(0).size() - 1;
            int outputSize = set.numClasses;
            hlayers[0] = inputSize;
            hlayers[hlayers.length - 1] = outputSize;

            Network nw = new Network(hlayers);

            // Train the network
            int testSetNum = 0;
            for (ArrayList<ArrayList<String>> trainSet: set.dataSets.trainingSets
                 ) {
                int numExamples = 5000; // Tuning parameter, used to control min total examples to train on (examples can be used more than once)
                while (numExamples > 0) {
                    for (ArrayList<String> example: trainSet
                         ) {
                        nw.initializeInputLayer(example);
                        nw.feedForward();
                        nw.backprop();
                        numExamples--;
                    }
                }
                // Test the network
                ArrayList<ArrayList<String>> testSet = set.dataSets.testSets.get(testSetNum);
                testSetNum++;
                // Keep track of results
                int correct = 0;
                int total = 0;
                for (ArrayList<String> test: testSet
                ) {
                    nw.initializeInputLayer(test);
                    nw.feedForward();
                    if (nw.guessedCorrectly())
                        correct++;
                    total++;
                }
                System.out.println((double) correct / total * 100);
            }
        }
        /*
        Network testNet = new Network(new int[] {6, 6, 4});
        System.out.println(testNet.getClassNumber());

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
        }*/
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