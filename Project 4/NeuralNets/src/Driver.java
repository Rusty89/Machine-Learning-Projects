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

    public static void main(String args[])throws Exception {

        // read in our categorical sets
        Data car = new CarData(new File("../DataSets/car.data"));
        Data abalone = new AbaloneData(new File("../DataSets/abalone.data"));
        Data segmentation = new ImageData(new File("../DataSets/segmentation.data"));

        // read in our regression sets, use regression and euclidean parameters for all these sets
        Data forestFire = new FireData(new File("../DataSets/forestfires.data"));
        Data machine = new MachineData(new File("../DataSets/machine.data"));
        Data redWine = new WineData(new File("../DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("../DataSets/winequality-white.csv"));

        // ------------------------------- begin driver for MLP Network -------------------------------
        ArrayList<Data> cData = new ArrayList<>();
        ArrayList<Data> rData = new ArrayList<>();

        // add categorical sets for our MLP network
        //cData.add(car);
        //cData.add(abalone);
        cData.add(segmentation);

        // add regression sets for our MLP network
        rData.add(forestFire);
        rData.add(machine);
        rData.add(redWine);
        rData.add(whiteWine);

        /*
        double bpLearningRate = 0.3;
        System.out.println("\n    Starting MLP using a learning rate of " + bpLearningRate);
        // start creating the MLP networks
        for (int hl = 0; hl < 3; hl++) {
            System.out.println("\nUsing " + hl + " hidden layers.");
            for (Data set : cData) {
                System.out.print("Running on data set " + set + ". Hidden nodes:");

                int inputSize = set.dataSets.trainingSets.get(0).get(0).size() - 1;
                int outputSize = set.numClasses;

                int[] hlayers = new int[hl + 2];
                for (int i = 0; i < hl; i++) {
                    hlayers[i + 1] = inputSize;
                    System.out.print(" " + inputSize);
                }
                System.out.println();

                hlayers[0] = inputSize;
                hlayers[hlayers.length - 1] = outputSize;

                System.out.println("Training...");
                // Train the network
                int testSetNum = 0;
                ArrayList<Double> precision = new ArrayList<>();
                ArrayList<Double> recall = new ArrayList<>();
                ArrayList<Double> accuracy = new ArrayList<>();
                for (ArrayList<ArrayList<String>> trainSet : set.dataSets.trainingSets) {
                    Network nw = new Network(hlayers, bpLearningRate);
                    int numExamples = 10000; // Tuning parameter, used to control min total examples to train on (examples can be used more than once)
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
                System.out.print("Running on data set " + set + ". Hidden nodes:");

                int inputSize = set.dataSets.trainingSets.get(0).get(0).size() - 1;
                int outputSize = 1;

                int[] hlayers = new int[hl + 2];
                for (int i = 0; i < hl; i++) {
                    hlayers[i + 1] = inputSize;
                    System.out.print(" " + inputSize);
                }
                System.out.println();

                hlayers[0] = inputSize;
                hlayers[hlayers.length - 1] = outputSize;

                // Train the network
                System.out.println("Training...");
                int testSetNum = 0;
                ArrayList<Double> aeResults = new ArrayList<>();
                ArrayList<Double> mseResults = new ArrayList<>();
                for (ArrayList<ArrayList<String>> trainSet : set.dataSets.trainingSets) {
                    Network nw = new Network(hlayers, bpLearningRate);
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

         */

        // runs all the particle swarm tests
        runParticleSwarmTests(rData, cData, 50);



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

    // function to run all Particle Swarm tests
    private static void runParticleSwarmTests(ArrayList<Data> rData, ArrayList<Data> cData, int failureLimit) throws IOException {
        for (Data data : cData) {
            FileWriter filer = new FileWriter(data.toString() + "PSOresults.csv");
            PrintWriter printer = new PrintWriter(filer);
            System.out.println("Running PSO tests for " + data.toString());

            // testing on a classification sets
            ArrayList<int []> hiddenLayers = new ArrayList<>();
            int numFeatures = data.fullSet.get(0).size()-1;
            // creates the parameters to make 0, 1 and 2 hidden layers
            hiddenLayers.add(new int []{numFeatures,  data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, numFeatures, data.numClasses});
            // iterates over the 0, 1 and 2 hidden layers tests
            for (int [] sizes : hiddenLayers) {
                ParticleSwarm pSwarm = new ParticleSwarm(data);
                for (int i = 0; i < 10; i++) {
                    ArrayList<Double> results = pSwarm.runTest(numFeatures * 3, sizes, failureLimit,  false, i);
                    System.out.println(results);
                    printer.print(results.get(0) + ",");
                    printer.print(results.get(1) + ",");
                    printer.print(results.get(2) + ",");
                    printer.println();
                }
            }
            printer.close();
            filer.close();
        }

        for (Data data : rData) {
            // testing on a regression sets
            FileWriter filer = new FileWriter(data.toString() + "PSOresults.csv");
            PrintWriter printer = new PrintWriter(filer);
            System.out.println("Running PSO tests for " + data.toString());
            ArrayList<int []> hiddenLayers = new ArrayList<>();
            int numFeatures = data.fullSet.get(0).size()-1;
            // creates the parameters to make 0, 1 and 2 hidden layrs
            hiddenLayers.add(new int []{numFeatures,  data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, numFeatures, data.numClasses});
            // iterates over the 0, 1 and 2 hidden layers tests
            for (int [] sizes : hiddenLayers) {
                ParticleSwarm pSwarm = new ParticleSwarm(data);
                for (int i = 0; i < 10; i++) {
                    ArrayList<Double> results = pSwarm.runTest(numFeatures * 3, sizes, failureLimit,  true, i);
                    System.out.println(results);
                    printer.print(results.get(0) +",");
                    printer.print(results.get(1) +",");
                    printer.println();
                }
            }
            printer.close();
            filer.close();
        }
    }
}