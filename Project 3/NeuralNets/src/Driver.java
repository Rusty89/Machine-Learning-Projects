/* A class that holds our main method for running our algorithms on the dataset. It uses the methods of the Data class
    to perform the algorithms on all data sets after reading in the data from files in the directory.
 */

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
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
        Data car = new CarData(new File("Project 3/DataSets/car.data"));
        Data abalone = new AbaloneData(new File("Project 3/DataSets/abalone.data"));
        Data segmentation = new ImageData(new File("Project 3/DataSets/segmentation.data"));

        // read in our regression sets, use regression and euclidean parameters for all these sets
        Data forestFire = new FireData(new File("Project 3/DataSets/forestfires.data"));
        Data machine = new MachineData(new File("Project 3/DataSets/machine.data"));
        Data redWine = new WineData(new File("Project 3/DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("Project 3/DataSets/winequality-white.csv"));

        ArrayList<Data> cData = new ArrayList<>();
        ArrayList<Data> rData = new ArrayList<>();

        // We can comment out which datasets we want to run here for video
        // Categorical
        cData.add(car);
        cData.add(abalone);
        cData.add(segmentation);
        // Regression
        rData.add(forestFire);
        rData.add(machine);
        rData.add(redWine);
        rData.add(whiteWine);

        for (int hl = 0; hl < 3; hl++) {
            System.out.println("\nUsing " + hl + " hidden layers.");
            for (Data set : cData) {
                // User input for setting up network
                System.out.println("Running on data set " + set);
                //System.out.print("How many hidden layers? ");
                //Scanner in = new Scanner(System.in);
                //String input = in.nextLine();
                //int hl = Integer.parseInt(input);
                int inputSize = set.dataSets.trainingSets.get(0).get(0).size() - 1;
                int outputSize = set.numClasses;

                int[] hlayers = new int[hl + 2];
                for (int i = 0; i < hl; i++) {
                    //System.out.print("How many nodes in hidden layer " + (i + 1) + "? ");
                    //input = in.nextLine();
                    //hlayers[i + 1] = Integer.parseInt(input);
                    hlayers[i + 1] = inputSize;
                }


                hlayers[0] = inputSize;
                hlayers[hlayers.length - 1] = outputSize;

                //System.out.println("Training...");
                // Train the network
                int testSetNum = 0;
                ArrayList<Double> precision = new ArrayList<>();
                ArrayList<Double> recall = new ArrayList<>();
                ArrayList<Double> accuracy = new ArrayList<>();
                for (ArrayList<ArrayList<String>> trainSet : set.dataSets.trainingSets) {
                    Network nw = new Network(hlayers);
                    int numExamples = 100000; // Tuning parameter, used to control min total examples to train on (examples can be used more than once)
                    while (numExamples > 0) {
                        for (ArrayList<String> example : trainSet) {
                            nw.initializeInputLayer(example);
                            nw.feedForward();
                            nw.cBackprop();
                            numExamples--;
                        }
                    }
                    // Test the network
                    ArrayList<ArrayList<String>> testSet = set.dataSets.testSets.get(testSetNum);
                    testSetNum++;
                    for (ArrayList<String> test : testSet) {
                        nw.initializeInputLayer(test);
                        nw.feedForward();
                        nw.guessHistory.add(nw.getClassNumber() + "");
                    }

                    ArrayList<String> loss = MathFunction.processConfusionMatrix(nw.guessHistory, testSet);
                    precision.add(Double.parseDouble(loss.get(0)));
                    recall.add(Double.parseDouble(loss.get(1)));
                    accuracy.add(Double.parseDouble(loss.get(2)));
                }
                printRangeAndMean("Precision", precision);
                printRangeAndMean("Recall", recall);
                printRangeAndMean("Accuracy", accuracy);
            }

            for (Data set : rData) {
                // User input for setting up network
                System.out.println("Running on data set " + set);
                //System.out.print("How many hidden layers? ");
                //Scanner in = new Scanner(System.in);
                //String input = in.nextLine();
                //int hl = Integer.parseInt(input);
                int inputSize = set.dataSets.trainingSets.get(0).get(0).size() - 1;
                int outputSize = 1;

                int[] hlayers = new int[hl + 2];
                for (int i = 0; i < hl; i++) {
                    //System.out.print("How many nodes in hidden layer " + (i + 1) + "? ");
                    //input = in.nextLine();
                    //hlayers[i + 1] = Integer.parseInt(input);
                    hlayers[i + 1] = inputSize;
                }

                hlayers[0] = inputSize;
                hlayers[hlayers.length - 1] = outputSize;

                // Train the network
                //System.out.println("Training...");
                int testSetNum = 0;
                ArrayList<Double> aeResults = new ArrayList<>();
                ArrayList<Double> mseResults = new ArrayList<>();
                for (ArrayList<ArrayList<String>> trainSet : set.dataSets.trainingSets) {
                    Network nw = new Network(hlayers);
                    int numExamples = 10000; // Tuning parameter, used to control min total examples to train on (examples can be used more than once)
                    while (numExamples > 0) {
                        for (ArrayList<String> example : trainSet) {
                            nw.initializeInputLayer(example);
                            nw.feedForward();
                            nw.rBackprop();
                            numExamples--;
                        }
                    }
                    // Test the network
                    ArrayList<ArrayList<String>> testSet = set.dataSets.testSets.get(testSetNum);
                    testSetNum++;
                    double absErr = 0;
                    double meanErr = 0;
                    for (ArrayList<String> test : testSet) {
                        nw.initializeInputLayer(test);
                        nw.feedForward();
                        absErr += Math.abs(nw.error);
                        meanErr += Math.pow(nw.error, 2);
                    }
                    absErr /= testSet.size();
                    meanErr /= testSet.size();
                    aeResults.add(absErr);
                    mseResults.add(meanErr);
                }
                printRangeAndMean("Absolute error", aeResults);
                printRangeAndMean("Mean squared error", mseResults);
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