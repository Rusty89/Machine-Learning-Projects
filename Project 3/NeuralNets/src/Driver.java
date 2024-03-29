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
        Data car = new CarData(new File("DataSets/car.data"));
        Data abalone = new AbaloneData(new File("DataSets/abalone.data"));
        Data segmentation = new ImageData(new File("DataSets/segmentation.data"));

        // read in our regression sets, use regression and euclidean parameters for all these sets
        Data forestFire = new FireData(new File("DataSets/forestfires.data"));
        Data machine = new MachineData(new File("DataSets/machine.data"));
        Data redWine = new WineData(new File("DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("DataSets/winequality-white.csv"));

        // ------------------------------- begin driver for MLP Network -------------------------------
        ArrayList<Data> cData = new ArrayList<>();
        ArrayList<Data> rData = new ArrayList<>();

        // add categorical sets for our MLP network
        cData.add(car);
        cData.add(abalone);
        cData.add(segmentation);

        // add regression sets for our MLP network
        rData.add(forestFire);
        rData.add(machine);
        rData.add(redWine);
        rData.add(whiteWine);

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


        // ---------------------------- start driving RBFN algorithm ----------------------------
        double learningRate = .5;
        System.out.println("\n\n    Starting RBF with a learning rate of " + learningRate);
        // run RBF for each data set
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> carCondensedTrainingSets = condenseData(car, false, false);
        ArrayList <RBFNetwork> RBFcar = makeRBFNetworks(car, carCondensedTrainingSets, true);
        trainRBFNetworks(car, RBFcar, learningRate, true);
        runRBFTests(car, RBFcar, true);

        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> abaloneCondensedTrainingSets = condenseData(abalone, false, true);
        ArrayList <RBFNetwork> RBFAbalone = makeRBFNetworks(abalone, abaloneCondensedTrainingSets, true);
        trainRBFNetworks(abalone, RBFAbalone, learningRate, true);
        runRBFTests(abalone, RBFAbalone, true);

        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> segmentationCondensedTrainingSets = condenseData(segmentation, false, true);
        ArrayList <RBFNetwork> RBFsegmetation = makeRBFNetworks(segmentation, segmentationCondensedTrainingSets, true);
        trainRBFNetworks(segmentation, RBFsegmetation, learningRate, true);
        runRBFTests(segmentation, RBFsegmetation, true);

        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> machineCondensedTrainingSets = condenseData(machine, true, true);
        ArrayList <RBFNetwork> RBFmachine = makeRBFNetworks(machine, machineCondensedTrainingSets, false);
        trainRBFNetworks(machine, RBFmachine, learningRate, false);
        runRBFTests(machine, RBFmachine, false);

        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> forestFireCondensedTrainingSets = condenseData(forestFire, true, true);
        ArrayList <RBFNetwork> RBFforestFire = makeRBFNetworks(forestFire, forestFireCondensedTrainingSets, false);
        trainRBFNetworks(forestFire, RBFforestFire, learningRate, false);
        runRBFTests(forestFire, RBFforestFire, false);

        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> redWineCondensedTrainingSets = condenseData(redWine, true, true);
        ArrayList <RBFNetwork> RBFredWine = makeRBFNetworks(redWine, redWineCondensedTrainingSets, false);
        trainRBFNetworks(redWine, RBFredWine, learningRate, false);
        runRBFTests(redWine, RBFredWine, false);

        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> whiteWineCondensedTrainingSets = condenseData(whiteWine, true, true);
        ArrayList <RBFNetwork> RBFwhiteWine = makeRBFNetworks(whiteWine, whiteWineCondensedTrainingSets, false);
        trainRBFNetworks(whiteWine, RBFwhiteWine, learningRate, false);
        runRBFTests(whiteWine,RBFwhiteWine, false);

    }

    // runs tests for our Radial Basis Function Network
    public static void runRBFTests(Data dataset, ArrayList<RBFNetwork> networks, boolean categorical) throws IOException {

        FileWriter filer = new FileWriter(dataset.toString() + "RBFresults.csv");
        PrintWriter printer = new PrintWriter(filer);

        System.out.println("Running RBF tests for " + dataset.toString());

        // if our data is categorical
        if(categorical) {
            for (int i = 0; i < numCondensedSets; i++) {
                System.out.println("\nRunning RBF tests for condensed set #" + (i + 1));
                for (int j = 0; j < numCVSSets; j++) {
                    int indexOfNetwork = (i * numCVSSets) + j;
                    ArrayList<String> predictions = new ArrayList<>();
                    for (int k = 0; k < dataset.dataSets.testSets.get(j).size(); k++) {
                        double predicted = networks.get(indexOfNetwork).classifyRBF(dataset.dataSets.testSets.get(j).get(k), categorical);
                        predictions.add((int)predicted + "");
                        System.out.println("\nPredicted value: " + predicted + ", values: " + dataset.dataSets.testSets.get(j).get(k));
                    }
                    ArrayList<String> results = MathFunction.processConfusionMatrix(predictions, dataset.dataSets.testSets.get(j));
                    System.out.println("\n-------------------------------------------");
                    System.out.println("Precision: " + results.get(0));
                    System.out.println("Recall: " + results.get(1));
                    System.out.println("Accuracy: " + results.get(2));
                    System.out.println("-------------------------------------------");
                    printer.print(results.get(0) + ",");
                    printer.print(results.get(1) + ",");
                    printer.print(results.get(2) + ",");
                    printer.println();
                }
            }
        }

        // for our regression sets
        else {
            // no condensed on regression
            for (int i = 0; i < numCondensedSets - 1; i++) {
                System.out.println("\nRunning RBF tests for condensed set# " + (i + 1));
                for (int j = 0; j < numCVSSets; j++) {
                    int indexOfNetwork = (i * numCVSSets) + j;
                    ArrayList<String> predictions = new ArrayList<>();
                    for (int k = 0; k < dataset.dataSets.testSets.get(j).size(); k++) {
                        double predicted = networks.get(indexOfNetwork).classifyRBF(dataset.dataSets.testSets.get(j).get(k), categorical);
                        predictions.add((int)predicted + "");
                        System.out.println(predicted + ": " + dataset.dataSets.testSets.get(j).get(k));
                    }
                    ArrayList<String> results = new ArrayList<>();
                    results.add(MathFunction.rootMeanSquaredError(predictions, dataset.dataSets.testSets.get(j), dataset.fullSet));
                    results.add(MathFunction.meanAbsoluteError(predictions, dataset.dataSets.testSets.get(j), dataset.fullSet));
                    System.out.println(results);
                    System.out.println("Mean Absolute Error: " + results.get(0));
                    System.out.println("Root Mean Squared Error: " + results.get(1));
                    printer.print(results.get(0) + ",");
                    printer.print(results.get(1) + ",");
                    printer.println();
                }
            }
        }
        printer.close();
        filer.close();
    }


    // function to tell the networks to train
    public static void trainRBFNetworks(Data dataset, ArrayList<RBFNetwork> networks, double learningRate, boolean categorical){
        if(categorical){
            for (int i = 0; i < numCondensedSets; i++) {
                System.out.println("\nTraining networks for condensed set #" + (i + 1));
                for (int j = 0; j < numCVSSets; j++) {
                    System.out.println("Training CVS set #" + (j + 1) + " of " + dataset.toString() + " with learning rate: " + learningRate);
                    int indexOfNetwork = (i * numCVSSets) + j;
                    networks.get(indexOfNetwork).trainRBFNetwork(dataset.dataSets.trainingSets.get(j), dataset.fullSet, learningRate, categorical);
                }
            }
        } else {
            for (int i = 0; i < numCondensedSets - 1; i++) {
                System.out.println("\nTraining networks for condensed set #" + (i + 1));
                for (int j = 0; j < numCVSSets ; j++) {
                    System.out.println("Training CVS set #" + (j + 1) + " of " + dataset.toString() + " with learning rate: " + learningRate);
                    int indexOfNetwork = (i * numCVSSets) + j;
                    networks.get(indexOfNetwork).trainRBFNetwork(dataset.dataSets.trainingSets.get(j), dataset.fullSet, learningRate, categorical);
                }
            }
        }
    }

    // constructs the RBF networks
    public static ArrayList<RBFNetwork> makeRBFNetworks(Data dataset, ArrayList<ArrayList<ArrayList<ArrayList<String>>>> condensedSets, boolean categorical) {

        // making the Radial Basis Networks for categorical outputs
        if(categorical){
            ArrayList<RBFNetwork> RBNetworks = new ArrayList<>();
            final int numFeatures = dataset.dataSets.trainingSets.get(0).get(0).size() - 1; // num features will not change with different training sets
            final int possibleOutcomes = dataset.numClassifications; // number of possible classifications in a data set

            // iterate over the three condensed sets: C-KNN, Kmeans, Kpam
            for (int i = 0; i < condensedSets.size(); i++) {
                System.out.println("\nMaking networks for condensed set #" + (i + 1));
                // iterate over the ten training sets for each dataset
                for (int j = 0; j < condensedSets.get(i).size(); j++) {
                    int condensedSetSize = condensedSets.get(i).get(j).size(); // get size of this particular condensed set
                    int[] layerSizes = {numFeatures, condensedSetSize, possibleOutcomes}; // so we know how many nodes go in different layers
                    RBFNetwork n = new RBFNetwork(layerSizes, condensedSets.get(i).get(j)); // build a network
                    RBNetworks.add(n); // add to Radial Basis networks array for later use
                    System.out.println("Making network #" + (j + 1) + " with " + layerSizes[1] + " nodes in the hidden layer.");
                }
            }

            return RBNetworks;


        } else {
            ArrayList<RBFNetwork> RBNetworks = new ArrayList<>();
            final int numFeatures = dataset.dataSets.trainingSets.get(0).get(0).size() - 1; // num features will not change with different training sets
            final int possibleOutcomes = 1; // number of possible classifications in a data set

            // iterate over the two condensed sets:  Kmeans, Kpam
            for (int i = 1; i < condensedSets.size(); i++) {
                System.out.println("\nMaking networks for condensed set #" + (i + 1));
                // iterate over the ten training sets for each dataset
                for (int j = 0; j < condensedSets.get(i).size(); j++) {
                    int condensedSetSize = condensedSets.get(i).get(j).size(); // get size of this particular condensed set
                    int[] layerSizes = {numFeatures, condensedSetSize, possibleOutcomes}; // so we know how many nodes go in different layers
                    RBFNetwork n = new RBFNetwork(layerSizes, condensedSets.get(i).get(j)); // build a network
                    RBNetworks.add(n); // add to Radial Basis networks array for later use
                    System.out.println("Making network #" + (j + 1) + " with " + layerSizes[0] + " nodes in the hidden layer.");
                }
            }
            return RBNetworks;
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

    // method to print a dataset to verify correct pre-processing and compilation
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