/* The Data Class is a parent to inheriting classes that are defined for each data set. It handles reading in files, pre-processing
    normalizing data, bucketizing for 10-fold cross validation, and methods for running each of the KNNAlgorithms.
 */

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class Data {
    public int numClasses;
    private static PrintWriter printer;
    public ArrayList<ArrayList<String>> fullSet = new ArrayList<>();
    public CVS dataSets = new CVS();
    private final int numTrainingSets = 10; // defines training sets for 10-fold cross validation
    public int numClassifications = 0;
    private ArrayList<ArrayList<ArrayList<String>>> condensedSet = new ArrayList<>(); // full condensed set from after cross-validation
    private ArrayList<ArrayList<ArrayList<String>>> kMeansSet = new ArrayList<>(); // full condensed set from after cross-validation
    private ArrayList<ArrayList<ArrayList<String>>> kPAMSet = new ArrayList<>(); // full condensed set from after cross-validation
    // private DecimalFormat df = new DecimalFormat("#.###");

    public ArrayList<ArrayList<ArrayList<String>>> getCondensedSet() {
        return condensedSet;
    }

    public ArrayList<ArrayList<ArrayList<String>>> getKMeansSet() {
        return kMeansSet;
    }

    public ArrayList<ArrayList<ArrayList<String>>> getKPAMSet() {
        return kPAMSet;
    }

    // drives the condensing of each data set
    public void condenseSets(boolean regression, boolean euclidean, String dataName) throws IOException {

        FileWriter filer = new FileWriter(dataName + " results.txt");
        printer = new PrintWriter(filer);

        // only run Condensed on classification (non-regression) sets
        if (!regression) {

            // begin our Condensed K-Nearest Neighbor Test, print to output and .txt
            System.out.println("Begin Condensed KNN condensing for " + dataName);
            printer.println("Begin Condensed KNN condensing for " + dataName);
            runCondensedKNN(euclidean);
            System.out.println("\nEnd test\n");
            printer.println("\nEnd test\n");

        }

        // begin our K-Means condensing, print to output and .txt
        System.out.println("Begin KMeans condensing for " + dataName);
        printer.println("Begin KMeans condensing for " + dataName);
        runKMeans(regression);
        System.out.println("\nEnd test\n");
        printer.println("\nEnd test\n");

        // begin our K Medoids PAM condensing, print to output and .txt
        System.out.println("Begin K Medoids PAM condensing for " + dataName);
        printer.println("Begin K Medoids PAM condensing for " + dataName);
        runKPAM(regression);
        System.out.println("\nEnd test\n");
        printer.println("\nEnd test\n");

        printer.close();
        filer.close();
    }

    // method to read in our data sets and convert them to an Java ArrayList for parsing
    public void fileTo2dStringArrayList(File inputFile) throws Exception {

        final int maxExamplesToRun = 300000; // max number of lines of data, to keep test manageable

        Scanner sc = new Scanner(inputFile); // read in our input file as an array list

        System.out.println("Reading in the " + inputFile.getName() + " and converting to an ArrayList");

        // convert dataset to an ArrayList
        while (sc.hasNextLine()) {
            ArrayList<String> line = new ArrayList<>(Arrays.asList(sc.nextLine().split(",")));
            ;
            fullSet.add(line);
        }

        // shuffle the input data into a random order
        System.out.println("Randomly shuffling the order of the data.");
        Collections.shuffle(fullSet);

        // remove data points if we don't want to run the full data set
        while (fullSet.size() > maxExamplesToRun) {
            fullSet.remove(0);
        }

    }

    // normalize continuous data to a range between 0-1
    public void normalizeData() {

        // local variables
        ArrayList<Double> max = new ArrayList<Double>();
        ArrayList<Double> min = new ArrayList<Double>();
        int indexOfLastTrait = fullSet.get(0).size() - 1;
        int sizeOfSet = fullSet.size();

        // find the maximum and minimum value of the original data sets
        for (int i = 0; i < indexOfLastTrait; i++) {
            max.add(Double.MIN_VALUE);
            min.add(Double.MAX_VALUE);
        }

        // iterate through each column of the dataset
        for (int j = 0; j < indexOfLastTrait; j++) {
            for (int i = 0; i < sizeOfSet; i++) {

                // find max and min values in each column of dataset
                max.set(j, Double.max(max.get(j), Double.parseDouble(fullSet.get(i).get(j))));
                min.set(j, Double.min(min.get(j), Double.parseDouble(fullSet.get(i).get(j))));
            }
        }

        // go through the dataset and normalize values between 0-1
        System.out.println("Normalizing Data between values 0 and 1.");
        for (int i = 0; i < sizeOfSet; i++) {
            for (int j = 0; j < indexOfLastTrait; j++) {

                // if max and min are the same, normalize it to a 1
                if ((max.get(j) - min.get(j)) == 0) {
                    fullSet.get(i).set(j, "1.0");
                } else {

                    // normalize the value using equation (x-min) / (max-min)
                    double normalizedValue = (Double.parseDouble(fullSet.get(i).get(j)) - min.get(j)) / (max.get(j) - min.get(j));
                    fullSet.get(i).set(j, normalizedValue + "");
                }
            }
        }
    }

    // bucketize the data for 10-fold cross validation
    public void bucketize() {

        // local constants
        double eightyPercentOfData = 0.8 * fullSet.size();
        double twentyPercentOfData = 0.2 * fullSet.size();
        double tenPercentOfData = 0.1 * fullSet.size();

        // local variables
        int countTrainingSet = 0;
        int countValidationAndTest;

        // iterate through the desired number of "buckets"
        System.out.println("Bucketizing the data for 10-fold cross validation.\n");
        for (int i = 0; i < numTrainingSets; i++) {

            // initializes new ArrayLists to store sets in the CVS structure
            dataSets.trainingSets.add(new ArrayList<ArrayList<String>>());
            dataSets.validationSets.add(new ArrayList<ArrayList<String>>());
            dataSets.testSets.add(new ArrayList<ArrayList<String>>());

            // generates a training set with 80% of the data
            for (int j = 0; j < eightyPercentOfData; j++) {

                // check to make the size of our training set is not larger than the size of the entire data set
                if (countTrainingSet < fullSet.size()) {
                    dataSets.trainingSets.get(i).add(fullSet.get(countTrainingSet));
                    countTrainingSet++;
                } else {
                    countTrainingSet = 0;
                    dataSets.trainingSets.get(i).add(fullSet.get(countTrainingSet));
                    countTrainingSet++;
                }
            }

            // sets the size of our validation and test sets to remaining data in training set
            countValidationAndTest = countTrainingSet;
            for (int j = 0; j < twentyPercentOfData; j++) {

                // generates validation set with the next 10% of data
                if (j < tenPercentOfData) {
                    if (countValidationAndTest < fullSet.size()) {
                        dataSets.validationSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    } else {
                        countValidationAndTest = 0;
                        dataSets.validationSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    }
                }

                // generates test sets with the last 10% of data
                else {
                    if (countValidationAndTest < fullSet.size()) {
                        dataSets.testSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    } else {
                        countValidationAndTest = 0;
                        dataSets.testSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    }
                }
            }
        }
    }

    public void findNumClassifications() {
        ArrayList<String>possibleOutcomes= new ArrayList<>();
        for (int i = 0; i < fullSet.size(); i++) {
            possibleOutcomes.add(fullSet.get(i).get(fullSet.get(0).size() - 1));
        }

        Set<String> set = new HashSet<String>(possibleOutcomes);
        this.numClassifications = set.size();

    }

    // drives the running of the Condensed K-Nearest Neighbor Algorithm
    public void runCondensedKNN(boolean euclidean) {

        int k = 1; // k with value 1 performed the best on most of our data sets

        System.out.println("\n---");
        System.out.println("Performing the algorithm with 10-fold cross validation for k = " + k + "\n");

        for (int i = 0; i < numTrainingSets; i++) {

            // actually runs condensed KNN condensing
            System.out.println("Condensing the training set " + (i + 1));
            ArrayList<ArrayList<String>> condensedSubset = KNNAlgorithms.CondensedKNN(dataSets.trainingSets.get(i), euclidean);
            condensedSet.add(condensedSubset); // add subset to the final condensed set to be passed into network
            System.out.println();
        }
    }

    // driver for K-Means algorithm 
    public void runKMeans(boolean regression) {

        System.out.println("\n---");
        System.out.println("Performing the algorithm with 10-fold cross validation\n");

        if (regression) {
            for (int i = 0; i < numTrainingSets; i++) {

                int numClusters = dataSets.trainingSets.get(i).size() / 4; // only uses n / 4 as per instruction

                // actually runs the KNNAlgorithms
                System.out.println("Running K-Means on the training set " + (i + 1));
                ArrayList<ArrayList<String>> KmeansSubset = KNNAlgorithms.Kmeans(dataSets.trainingSets.get(i), numClusters);
                kMeansSet.add(KmeansSubset); // add subset to the final condensed set to be passed into network
                System.out.println();
            }

        }

        else {

            for (int i = 0; i < numTrainingSets; i++) {
                int numClusters = condensedSet.get(i).size();

                // actually runs the KNNAlgorithms
                System.out.println("Running K-Means on the training set " + (i + 1));
                ArrayList<ArrayList<String>> KmeansSubset = KNNAlgorithms.Kmeans(dataSets.trainingSets.get(i), numClusters);
                kMeansSet.add(KmeansSubset); // add subset to the final condensed set to be passed into network
                System.out.println();
            }
        }
    }

    // driver for Partioning Around Medoids algorithm
    public void runKPAM(boolean regression) {

        System.out.println("\n---");
        System.out.println("Performing the algorithm with 10-fold cross validation\n");

        if (regression) {

            for (int i = 0; i < numTrainingSets; i++) {

                int numClusters = dataSets.trainingSets.get(i).size() / 4; // only uses n / 4 as per instruction

                // actually runs the KNNAlgorithms
                System.out.println("Running PAM on the training set " + (i + 1));
                ArrayList<ArrayList<String>> KPAMSubset = KNNAlgorithms.AlternativePAM(dataSets.trainingSets.get(i), numClusters);
                kPAMSet.add(KPAMSubset); // add subset to the final condensed set to be passed into network
                System.out.println();
            }
        }

        else {

            for (int i = 0; i < numTrainingSets; i++) {
                int numClusters = condensedSet.get(i).size();

                // actually runs the KNNAlgorithms
                System.out.println("Running PAM on the training set " + (i + 1));
                ArrayList<ArrayList<String>> KPAMSubset = KNNAlgorithms.AlternativePAM(dataSets.trainingSets.get(i), numClusters);
                kPAMSet.add(KPAMSubset); // add subset to the final condensed set to be passed into network
                System.out.println();
            }
        }
    }
}
