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

        ArrayList<Data> cData = new ArrayList<>();
        ArrayList<Data> rData = new ArrayList<>();
        // read in our categorical sets
        Data car = new CarData(new File("../../Machine-Learning-Projects/DataSets/car.data"));
        Data abalone = new AbaloneData(new File("../../Machine-Learning-Projects/DataSets/abalone.data"));
        Data segmentation = new ImageData(new File("../../Machine-Learning-Projects/DataSets/segmentation.data"));

        // read in our regression sets, use regression and euclidean parameters for all these sets
        Data forestFire = new FireData(new File("../../Machine-Learning-Projects/DataSets/forestfires.data"));
        Data machine = new MachineData(new File("../../Machine-Learning-Projects/DataSets/machine.data"));
        Data redWine = new WineData(new File("../../Machine-Learning-Projects/DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("../../Machine-Learning-Projects/DataSets/winequality-white.csv"));

        cData.add(car);
        cData.add(abalone);
        cData.add(segmentation);

        rData.add(forestFire);
        rData.add(machine);
        rData.add(redWine);
        rData.add(whiteWine);

        for (Data d : cData) {
            for (int i = 0; i < 3; i++) {
                int[] layers = new int[2 + i];
                for (int j = 0; j < layers.length; j++) {
                    if (j == layers.length - 1) {
                        layers[j] = d.numClasses;
                    } else {
                        layers[j] = d.dataSets.trainingSets.get(0).get(0).size() - 1;
                    }
                }
                GeneticAlgorithm GA = new GeneticAlgorithm(d.dataSets.trainingSets.get(0), layers, d.numClasses, false);
                // test on 10 generations for each training set
                for (ArrayList<ArrayList<String>> trainingSet : d.dataSets.trainingSets) {
                    GA.trainSet = trainingSet;
                    for (int k = 0; k < 5; k++) {
                        GA.runAlgorithm();
                    }
                }
                // Grab best GA
                Network bestGA = GA.best;

                // calculate results
                ArrayList<Double> GAprecision = new ArrayList<>();
                ArrayList<Double> GArecall = new ArrayList<>();
                ArrayList<Double> GAaccuracy = new ArrayList<>();
                for (ArrayList<ArrayList<String>> testSet : d.dataSets.testSets) {
                    for (ArrayList<String> test : testSet) {
                        bestGA.initializeInputLayer(test);
                        bestGA.feedForward();
                        bestGA.guessHistory.add(bestGA.getClassNumber() + "");
                    }
                    ArrayList<String> GAloss = MathFunction.processConfusionMatrix(bestGA.guessHistory, testSet);
                    bestGA.guessHistory.clear();
                    GAprecision.add(Double.parseDouble(GAloss.get(0)));
                    GArecall.add(Double.parseDouble(GAloss.get(1)));
                    GAaccuracy.add(Double.parseDouble(GAloss.get(2)));
                }
                System.out.println("Hidden Layers = " + i);
                printRangeAndMean(d.toString() + " data - GAPrecision: ", GAprecision);
                printRangeAndMean(d.toString() + " data -    GARecall: ", GArecall);
                printRangeAndMean(d.toString() + " data -  GAAccuracy: ", GAaccuracy);
            }
        }

        for (Data d : rData) {
            for (int i = 0; i < 3; i++) {
                int[] layers = new int[2 + i];
                for (int j = 0; j < layers.length; j++) {
                    if (j == layers.length - 1) {
                        layers[j] = 1;
                    } else {
                        layers[j] = d.dataSets.trainingSets.get(0).get(0).size() - 1;
                    }
                }
                GeneticAlgorithm GA = new GeneticAlgorithm(d.dataSets.trainingSets.get(0), layers, 1, true);
                // test on 10 generations for each training set
                for (ArrayList<ArrayList<String>> trainingSet : d.dataSets.trainingSets) {
                    GA.trainSet = trainingSet;
                    for (int k = 0; k < 5; k++) {
                        GA.runAlgorithm();
                    }
                }
                // Grab best GA
                Network bestGA = GA.best;

                // calculate results
                ArrayList<Double> aeResults = new ArrayList<>();
                ArrayList<Double> mseResults = new ArrayList<>();
                for (ArrayList<ArrayList<String>> testSet : d.dataSets.testSets) {
                    double absErr = 0;
                    double meanErr = 0;
                    for (ArrayList<String> test : testSet) {
                        bestGA.initializeInputLayer(test);
                        bestGA.feedForward();
                        bestGA.calcErr();
                        absErr += Math.abs(bestGA.error);
                        meanErr += Math.pow(bestGA.error, 2);
                    }
                    absErr /= testSet.size();
                    meanErr /= testSet.size();
                    aeResults.add(absErr);
                    mseResults.add(meanErr);
                }
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