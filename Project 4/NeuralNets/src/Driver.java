/* A class that holds our main method for running our algorithms on the dataset. It uses the methods of the Data class
    to perform the algorithms on all data sets after reading in the data from files in the directory.
 */

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.text.DecimalFormat;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

public class Driver {

    private static final int numCondensedSets = 3;
    private static final int numCVSSets = 1;

    public static void main(String[] args) throws Exception {

        // Process Data
        // Read in our categorical sets
        Data car = new CarData(new File("../../Machine-Learning-Projects/DataSets/car.data"));
        Data abalone = new AbaloneData(new File("../../Machine-Learning-Projects/DataSets/abalone.data"));
        Data segmentation = new ImageData(new File("../../Machine-Learning-Projects/DataSets/segmentation.data"));

        // Read in our regression sets, use regression and euclidean parameters for all these sets
        Data forestFire = new FireData(new File("../../Machine-Learning-Projects/DataSets/forestfires.data"));
        Data machine = new MachineData(new File("../../Machine-Learning-Projects/DataSets/machine.data"));
        Data redWine = new WineData(new File("../../Machine-Learning-Projects/DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("../../Machine-Learning-Projects/DataSets/winequality-white.csv"));

        // Store data by categorical and regression
        ArrayList<Data> cData = new ArrayList<>();
        ArrayList<Data> rData = new ArrayList<>();

        // Categorical
        cData.add(car);
        cData.add(abalone);
        cData.add(segmentation);

        // Regression
        rData.add(forestFire);
        rData.add(machine);
        rData.add(redWine);
        rData.add(whiteWine);

        // Train and test categorical data
        for (Data d : cData) {
            // For hidden layers 0, 1, and 2
            for (int i = 0; i < 3; i++) {
                int[] layers = new int[2 + i];
                for (int j = 0; j < layers.length; j++) {
                    if (j == layers.length - 1) {
                        // Last layer is output layer
                        layers[j] = d.numClasses;
                    } else {
                        // All layers except output are the same size, equal to the amount of features
                        layers[j] = d.dataSets.trainingSets.get(0).get(0).size() - 1;
                    }
                }
                // Initialize GA
                GeneticAlgorithm GA = new GeneticAlgorithm(d.dataSets.trainingSets.get(0), layers, d.numClasses, false);
                // Test on 10 generations for each training set - 100 generations total for training
                for (ArrayList<ArrayList<String>> trainingSet : d.dataSets.trainingSets) {
                    // Update training set
                    GA.trainSet = trainingSet;
                    for (int k = 0; k < 10; k++) {
                        GA.runAlgorithm();
                    }
                }
                // Grab best GA
                Network bestGA = GA.best;

                // Store results
                ArrayList<Double> GAprecision = new ArrayList<>();
                ArrayList<Double> GArecall = new ArrayList<>();
                ArrayList<Double> GAaccuracy = new ArrayList<>();
                // Test on all test sets
                for (ArrayList<ArrayList<String>> testSet : d.dataSets.testSets) {
                    // On each example within the set
                    for (ArrayList<String> test : testSet) {
                        bestGA.initializeInputLayer(test);
                        bestGA.feedForward();
                        bestGA.guessHistory.add(bestGA.getClassNumber() + "");
                    }
                    // Process results
                    ArrayList<String> GAloss = MathFunction.processConfusionMatrix(bestGA.guessHistory, testSet);
                    bestGA.guessHistory.clear();
                    GAprecision.add(Double.parseDouble(GAloss.get(0)));
                    GArecall.add(Double.parseDouble(GAloss.get(1)));
                    GAaccuracy.add(Double.parseDouble(GAloss.get(2)));
                }
                // Print results
                System.out.println("Hidden Layers = " + i);
                printRangeAndMean(d.toString() + " data - GAPrecision: ", GAprecision);
                printRangeAndMean(d.toString() + " data -    GARecall: ", GArecall);
                printRangeAndMean(d.toString() + " data -  GAAccuracy: ", GAaccuracy);
            }
        }

        // Regression data
        for (Data d : rData) {
            // For hidden layers 0, 1, 2
            for (int i = 0; i < 3; i++) {
                int[] layers = new int[2 + i];
                for (int j = 0; j < layers.length; j++) {
                    if (j == layers.length - 1) {
                        // Only 1 output node for regression
                        layers[j] = 1;
                    } else {
                        // All layers except output are the same size, equal to the number of features
                        layers[j] = d.dataSets.trainingSets.get(0).get(0).size() - 1;
                    }
                }
                // Initialize GA
                GeneticAlgorithm GA = new GeneticAlgorithm(d.dataSets.trainingSets.get(0), layers, 1, true);
                // Test on 10 generations for each training set - 100 generations total
                for (ArrayList<ArrayList<String>> trainingSet : d.dataSets.trainingSets) {
                    // Update training set
                    GA.trainSet = trainingSet;
                    for (int k = 0; k < 5; k++) {
                        GA.runAlgorithm();
                    }
                }
                // Grab best GA
                Network bestGA = GA.best;

                // Store results
                ArrayList<Double> aeResults = new ArrayList<>();
                ArrayList<Double> mseResults = new ArrayList<>();
                // Test on each test set
                for (ArrayList<ArrayList<String>> testSet : d.dataSets.testSets) {
                    double absErr = 0;
                    double meanErr = 0;
                    // On each example in the test set
                    for (ArrayList<String> test : testSet) {
                        bestGA.initializeInputLayer(test);
                        bestGA.feedForward();
                        bestGA.calcErr();
                        // Tally errors
                        absErr += Math.abs(bestGA.error);
                        meanErr += Math.pow(bestGA.error, 2);
                    }
                    // Calculate results
                    absErr /= testSet.size();
                    meanErr /= testSet.size();
                    aeResults.add(absErr);
                    mseResults.add(meanErr);
                }
                // Print Results
                System.out.println("Hidden Layers = " + i);
                printRangeAndMean(d.toString() + " data -     Absolute Error: ", aeResults);
                printRangeAndMean(d.toString() + " data - Mean Squared Error: ", mseResults);
            }
        }
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