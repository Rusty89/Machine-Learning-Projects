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
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Driver {

    private static final int oneTenthNumGenerations = 10;
    private static final int numCondensedSets = 3;
    private static final int numFailuresAllowed = 30;

    public static void main(String[] args) throws Exception {

        // Process Data
        // Read in our categorical sets
        Data car = new CarData(new File("DataSets/car.data"));
        Data abalone = new AbaloneData(new File("DataSets/abalone.data"));
        Data segmentation = new ImageData(new File("DataSets/segmentation.data"));

        // Read in our regression sets, use regression and euclidean parameters for all these sets
        Data forestFire = new FireData(new File("DataSets/forestfires.data"));
        Data machine = new MachineData(new File("DataSets/machine.data"));
        Data redWine = new WineData(new File("DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("DataSets/winequality-white.csv"));

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

        // run the tests for the three genetic aproaches
        runGeneticAlgorithmTests(rData, cData);
        runParticleSwarmTests(rData, cData, numFailuresAllowed);
        runDifferentialEvoltionTests(rData, cData);
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

    private static void runGeneticAlgorithmTests (ArrayList<Data> rData, ArrayList<Data> cData)
    {
        ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        Network[] best = new Network[3];

        // Train and test categorical data
        for (Data d : cData) {
            // For hidden layers 0, 1, and 2
            for (int i = 0; i < 3; i++) {
                int finalI = i;  //save i as a new int for the runnable to use
                executorService.execute(() -> {
                    int[] layers = new int[2 + finalI];
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
                        for (int k = 0; k < oneTenthNumGenerations; k++) {
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
                    System.out.println("Hidden Layers = " + finalI);
                    printRangeAndMean(d.toString() + " data - GAPrecision: ", GAprecision);
                    printRangeAndMean(d.toString() + " data -    GARecall: ", GArecall);
                    printRangeAndMean(d.toString() + " data -  GAAccuracy: ", GAaccuracy);

                    best[finalI] = bestGA;
                });
            }
        }

        // Regression data
        for (Data d : rData) {
            // For hidden layers 0, 1, 2
            for (int i = 0; i < 3; i++) {
                int finalI = i;
                executorService.execute(() -> {
                    int[] layers = new int[2 + finalI];
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
                        for (int k = 0; k < oneTenthNumGenerations; k++) {
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
                    System.out.println("Hidden Layers = " + finalI);
                    printRangeAndMean(d.toString() + " data -     Absolute Error: ", aeResults);
                    printRangeAndMean(d.toString() + " data - Mean Squared Error: ", mseResults);
                });
            }
        }

        executorService.shutdown();
        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
        }

        catch (InterruptedException e)
        {
            
        }
      
        for (Network goodOne: best) {
            System.out.println(goodOne);
        }
    }

    // function to run all Particle Swarm tests
    private static void runParticleSwarmTests(ArrayList<Data> rData, ArrayList<Data> cData, int failureLimit) throws IOException {
        for (Data data : cData) {

            // testing on a classification sets
            FileWriter filer = new FileWriter(data.toString() + "PSOresults.csv");
            PrintWriter printer = new PrintWriter(filer);
            System.out.println("Running PSO tests for " + data.toString());
            System.out.println("----------------------------------------");

            ArrayList<int []> hiddenLayers = new ArrayList<>();
            int numFeatures = data.fullSet.get(0).size() -  1;

            // creates the parameters to make 0, 1 and 2 hidden layers
            hiddenLayers.add(new int []{numFeatures,  data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, numFeatures, data.numClasses});

            // iterates over the 0, 1 and 2 hidden layers tests
            for (int i = 0; i < hiddenLayers.size(); i++) {
                System.out.println("\nHidden Layers = " + i);
                ParticleSwarm pSwarm = new ParticleSwarm(data);
                for (int j = 0; j < 10; j++) {
                    ArrayList<Double> results = pSwarm.runTest(numFeatures * numCondensedSets, hiddenLayers.get(i), failureLimit,  false, j);
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
            System.out.println("----------------------------------------");

            // testing on a classification sets
            ArrayList<int []> hiddenLayers = new ArrayList<>();
            int numFeatures = data.fullSet.get(0).size() - 1;

            // creates the parameters to make 0, 1 and 2 hidden layers
            hiddenLayers.add(new int []{numFeatures,  data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, numFeatures, data.numClasses});

            // iterates over the 0, 1 and 2 hidden layers tests
            for (int i = 0; i < hiddenLayers.size(); i++) {
                System.out.println("\nHidden Layers = " + i);
                ParticleSwarm pSwarm = new ParticleSwarm(data);
                for (int j = 0; j < 10; j++) {
                    ArrayList<Double> results = pSwarm.runTest(numFeatures * numCondensedSets, hiddenLayers.get(i), failureLimit,  true, j);
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

    // function to run all Differential Evolution
    private static void runDifferentialEvolutionTests(ArrayList<Data> rData, ArrayList<Data> cData) throws IOException {
        for (Data data : cData) {
            // testing on a classification sets
            FileWriter filer = new FileWriter(data.toString() + "DEresults.csv");
            PrintWriter printer = new PrintWriter(filer);
            System.out.println("Running DE tests for " + data.toString());
            System.out.println("----------------------------------------");

            ArrayList<int []> hiddenLayers = new ArrayList<>();
            int numFeatures = data.fullSet.get(0).size() - 1;

            // creates the parameters to make 0, 1 and 2 hidden layers
            hiddenLayers.add(new int []{numFeatures,  data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, numFeatures, data.numClasses});

            // iterates over the 0, 1 and 2 hidden layers tests
            for (int i = 0; i < hiddenLayers.size(); i++) {
                System.out.println("\nHidden Layers = " + i);
                DifferentialEvolution DE = new DifferentialEvolution(data);
                for (int j = 0; j < 10; j++) {
                    ArrayList<Double> results = DE.runTest(numFeatures * numCondensedSets, hiddenLayers.get(i),false, j);
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
            FileWriter filer = new FileWriter(data.toString() + "DEresults.csv");
            PrintWriter printer = new PrintWriter(filer);
            System.out.println("Running DE tests for " + data.toString());
            System.out.println("----------------------------------------");

            ArrayList<int []> hiddenLayers = new ArrayList<>();
            int numFeatures = data.fullSet.get(0).size() - 1;

            // creates the parameters to make 0, 1 and 2 hidden layers
            hiddenLayers.add(new int []{numFeatures,  data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, data.numClasses});
            hiddenLayers.add(new int []{numFeatures, numFeatures, numFeatures, data.numClasses});

            // iterates over the 0, 1 and 2 hidden layers tests
            for (int i = 0; i < hiddenLayers.size(); i++) {
                System.out.println("\nHidden Layers = " + i);
                DifferentialEvolution DE = new DifferentialEvolution(data);
                for (int j = 0; j < 10; j++) {
                    ArrayList<Double> results = DE.runTest(numFeatures * numCondensedSets, hiddenLayers.get(i), true, j);
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
