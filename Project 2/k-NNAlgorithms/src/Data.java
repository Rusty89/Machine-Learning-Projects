import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

public class Data {
    private static PrintWriter printer;
    public ArrayList<ArrayList<String>> fullSet = new ArrayList<>();
    public CVS dataSets = new CVS();
    private final int numTrainingSets = 10; // defines training sets for 10-fold cross validation
    private final int kValueSelections[] = {1, 3, 5}; // choose odd values for k to avoid tie-breakers
    private ArrayList<ArrayList<ArrayList<String>>> editedSets = new ArrayList<>();
    private ArrayList<ArrayList<ArrayList<String>>> condensedSets = new ArrayList<>();
    private DecimalFormat df = new DecimalFormat("#.###");

    // drives the running of each test
    public void runTests(boolean regression, boolean euclidean, String dataName) throws IOException {

        FileWriter filer = new FileWriter(dataName + " results.txt");
        printer = new PrintWriter(filer);

        // begin our K-Nearest Neighbor Test, print to output and .txt
        System.out.println("Begin KNN test for " + dataName);
        printer.println("Begin KNN test for " + dataName);
        runKNN(regression, euclidean);
        System.out.println("\nEnd test\n");
        printer.println("\nEnd test\n");

        // only run Condensed & Edited KNN on classification (non-regression) sets
        if(!regression){

            // begin our Condensed K-Nearest Neighbor Test, print to output and .txt
            System.out.println("Begin Condensed KNN test for " + dataName);
            printer.println("Begin Condensed KNN test for " + dataName);
            runCondensedKNN(euclidean);
            System.out.println("\nEnd test\n");
            printer.println("\nEnd test\n");

            // begin our Edited K-Nearest Neighbor Test, print to output and .txt
            System.out.println("Begin Edited KNN test for " + dataName);
            printer.println("Begin Edited KNN test for " + dataName);
            runEditedKNN(euclidean);
            System.out.println("\nEnd test\n");
            printer.println("\nEnd test\n");
        }

        // begin our K-Means Test, print to output and .txt
        System.out.println("Begin KMeans test for " + dataName);
        printer.println("Begin KMeans test for " + dataName);
        runKMeans(regression,euclidean);
        System.out.println("\nEnd test\n");
        printer.println("\nEnd test\n");

        // begin our K Medoids PAM Test, print to output and .txt
        System.out.println("Begin K Medoids PAM test for " + dataName);
        printer.println("Begin K Medoids PAM test for " + dataName);
        runKPAM(regression,euclidean);
        System.out.println("\nEnd test\n");
        printer.println("\nEnd test\n");

        printer.close();
        filer.close();
    }

    // method to read in our data sets and convert them to an Java ArrayList for parsing
    public void fileTo2dStringArrayList(File inputFile) throws Exception {

        final int maxExamplesToRun = 100; // max number of lines of data, to keep test manageable

        Scanner sc = new Scanner(inputFile); // read in our input file as an array list

        System.out.println("Reading in the " + inputFile.getName() + " and converting to an ArrayList");

        // convert dataset to an ArrayList
        while (sc.hasNextLine()){
            ArrayList<String> line= new ArrayList<>(Arrays.asList(sc.nextLine().split(",")));;
            fullSet.add(line);
        }

        // shuffle the input data into a random order
        System.out.println("Randomly shuffling the order of the data.");
        Collections.shuffle(fullSet);

        // remove data points if we don't want to run the full data set
        while(fullSet.size() > maxExamplesToRun){
            fullSet.remove(0);
        }

    }

    // normalize continuous data to a range between 0-1
    public void normalizeData(){

        // local variables
        ArrayList<Double> max = new ArrayList<Double>();
        ArrayList<Double> min = new ArrayList<Double>();
        int indexOfLastTrait = fullSet.get(0).size() - 1;
        int sizeOfSet = fullSet.size();

        // find the maximum and minimum value of the original data sets
        for (int i = 0; i < indexOfLastTrait ; i++) {
            max.add(Double.MIN_VALUE);
            min.add(Double.MAX_VALUE);
        }

        // iterate through each column of the dataset
        for (int j = 0; j < indexOfLastTrait; j++) {
            for (int i = 0; i < sizeOfSet ; i++) {

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
                if((max.get(j) - min.get(j)) == 0){
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
                }
                else{
                    countTrainingSet = 0;
                    dataSets.trainingSets.get(i).add(fullSet.get(countTrainingSet));
                    countTrainingSet++;
                }
            }

            // sets the size of our validation and test sets to remaining data in training set
            countValidationAndTest = countTrainingSet;
            for (int j = 0; j < twentyPercentOfData; j++) {

                // generates validation set with the next 10% of data
                if(j < tenPercentOfData){
                    if (countValidationAndTest < fullSet.size()) {
                        dataSets.validationSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    }
                    else{
                        countValidationAndTest = 0;
                        dataSets.validationSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    }
                }

                // generates test sets with the last 10% of data
                else{
                    if (countValidationAndTest < fullSet.size()) {
                        dataSets.testSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    }
                    else{
                        countValidationAndTest = 0;
                        dataSets.testSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    }
                }
            }
        }
    }

    // drives the running of the K-Nearest Neighbor Algorithm
    public void runKNN(boolean regression, boolean euclidean){

        // calculates and returns loss functions for regression data sets
        if(regression) {

            for (int k : kValueSelections) {
                System.out.println("\n---");
                System.out.println("Performing the algorithm with 10-fold cross validation for k = " + k + "\n");

                // calculate the root mean squared and absolute error
                double RMSE = 0;
                double absError = 0;
                for (int i = 0; i < numTrainingSets; i++) {

                    // actually runs the algorithm
                    System.out.println("Running KNN on training set " + (i + 1));
                    ArrayList<String> result1 = Algorithms.KNN(dataSets.trainingSets.get(i), dataSets.testSets.get(i), k, regression, euclidean);

                    // calculate the loss functions
                    absError += Double.parseDouble(MathFunction.meanAbsoluteError(result1, dataSets.testSets.get(i), fullSet));
                    RMSE += Double.parseDouble(MathFunction.rootMeanSquaredError(result1, dataSets.testSets.get(i), fullSet));

                    System.out.println();
                }

                System.out.println("Calculating regression loss functions for k = " + k);
                System.out.println("Mean Absolute error is : " + df.format(absError / numTrainingSets) + "  Root Mean Squared Error : " + df.format(RMSE / numTrainingSets));
                printer.println("Mean Absolute error is : " + df.format(absError / numTrainingSets) + "  Root Mean Squared Error : " + df.format(RMSE / numTrainingSets));
            }
        }

        // calculates loss functions for classification sets
        else {

            for (int k : kValueSelections) {
                System.out.println("\n---");
                System.out.println("Performing the algorithm with 10-fold cross validation for k = " + k + "\n");
                double precisionAvg = 0;
                double recallAvg = 0;
                double accuracyAvg = 0;

                for (int i = 0; i < numTrainingSets; i++) {

                    // actually runs the algorithm
                    System.out.println("Running KNN on training set " + (i + 1));
                    ArrayList<String>result=Algorithms.KNN(dataSets.trainingSets.get(i),dataSets.testSets.get(i), k, regression, euclidean);

                    // loss function calculations
                    result = MathFunction.processConfusionMatrix(result, dataSets.testSets.get(i));
                    precisionAvg += Double.parseDouble(result.get(0));
                    recallAvg += Double.parseDouble(result.get(1));
                    accuracyAvg += Double.parseDouble(result.get(2));

                    System.out.println();

                }
                System.out.println("Calculating classification loss functions for k = " + k);
                System.out.println("Precision is: " + df.format(precisionAvg / numTrainingSets) + "\nRecall is: " + df.format(recallAvg / numTrainingSets) + "\nAccuracy is: " + df.format(accuracyAvg / numTrainingSets));
                printer.println("Precision is: " + df.format(precisionAvg / numTrainingSets) + "\nRecall is: " + df.format(recallAvg / numTrainingSets) + "\nAccuracy is: " + df.format(accuracyAvg / numTrainingSets));
            }
        }
    }

    // drives the running of the Edited K-Nearest Neighbor Algorithm
    public void runEditedKNN(boolean euclidean){

        for (int k : kValueSelections) {
            System.out.println("\n---");
            System.out.println("Performing the algorithm with 10-fold cross validation for k = " + k + "\n");
            double precisionAvg = 0;
            double recallAvg = 0;
            double accuracyAvg = 0;

            for (int i = 0; i < numTrainingSets; i++) {

                // actually runs the algorithms
                System.out.println("Editing the training set " + (i + 1));
                ArrayList<ArrayList<String>> editedSet = Algorithms.EditedKNN(dataSets.trainingSets.get(i), dataSets.validationSets.get(i), k, false, true);
                System.out.println("Passing the edited data set " + (i + 1) + " into KNN");
                ArrayList<String>result=Algorithms.KNN(editedSet,dataSets.testSets.get(i), k, false, euclidean);

                // loss function calculations
                result = MathFunction.processConfusionMatrix(result, dataSets.testSets.get(i));
                precisionAvg += Double.parseDouble(result.get(0));
                recallAvg += Double.parseDouble(result.get(1));
                accuracyAvg += Double.parseDouble(result.get(2));

                editedSets.add(editedSet);
                System.out.println();

            }
            System.out.println("Calculating classification loss functions for k = " + k);
            System.out.println("Precision is: " + df.format(precisionAvg / numTrainingSets) + "\nRecall is: " + df.format(recallAvg / numTrainingSets) + "\nAccuracy is: " + df.format(accuracyAvg / numTrainingSets));
            printer.println("Precision is: " + df.format(precisionAvg / numTrainingSets) + "\nRecall is: " + df.format(recallAvg / numTrainingSets) + "\nAccuracy is: " + df.format(accuracyAvg / numTrainingSets));
        }
    }

    // drives the running of the Condensed K-Nearest Neighbor Algorithm
    public void runCondensedKNN( boolean euclidean){
        for (int k : kValueSelections) {
            System.out.println("\n---");
            System.out.println("Performing the algorithm with 10-fold cross validation for k = " + k + "\n");

            double precisionAvg = 0;
            double recallAvg = 0;
            double accuracyAvg = 0;

            for (int i = 0; i < numTrainingSets; i++) {

                // actually runs the algorithms
                System.out.println("Condensing the training set " + (i + 1));
                ArrayList<ArrayList<String>> condensedSet = Algorithms.CondensedKNN(dataSets.trainingSets.get(i), euclidean);
                System.out.println("Passing the condensed data set " + (i + 1) + " into KNN");
                ArrayList<String> result = Algorithms.KNN(condensedSet,dataSets.testSets.get(i), k, false, euclidean);

                // calculate loss functions
                result = MathFunction.processConfusionMatrix(result, dataSets.testSets.get(i));
                precisionAvg += Double.parseDouble(result.get(0));
                recallAvg += Double.parseDouble(result.get(1));
                accuracyAvg += Double.parseDouble(result.get(2));

                condensedSets.add(condensedSet);
                System.out.println();
            }
            System.out.println("Calculating classification loss functions for k = " + k);
            System.out.println("Precision is: " + df.format(precisionAvg / numTrainingSets) + "\nRecall is: " + df.format(recallAvg / numTrainingSets) + "\nAccuracy is: " + df.format(accuracyAvg / numTrainingSets));
            printer.println("Precision is: " + df.format(precisionAvg / numTrainingSets) + "\nRecall is: " + df.format(recallAvg / numTrainingSets) + "\nAccuracy is: " + df.format(accuracyAvg / numTrainingSets));
        }
    }

    // driver for K-Means algorithm 
    public void runKMeans(boolean regression, boolean euclidean) {

        System.out.println("\n---");
        System.out.println("Performing the algorithm with 10-fold cross validation\n");
        
        // calculates and returns loss functions for regression data sets
        if(regression){

            // calculate the root mean squared and absolute error
            double RMSE = 0; 
            double absError = 0;
            for (int i = 0; i < numTrainingSets; i++) {

                
                int numClusters = dataSets.trainingSets.size() / 4; // only uses n / 4 as per instruction

                // actually runs the algorithms
                System.out.println("Running K-Means on the training set " + (i + 1));
                ArrayList<ArrayList<String>> KmeansSet = Algorithms.Kmeans(dataSets.trainingSets.get(i), numClusters );
                System.out.println("Passing the condensed data set " + (i + 1) + " into KNN");
                ArrayList<String> result1 = Algorithms.KNN(KmeansSet, dataSets.testSets.get(i), 1, regression, euclidean);

                absError += Double.parseDouble(MathFunction.meanAbsoluteError(result1, dataSets.testSets.get(i), fullSet));
                RMSE += Double.parseDouble(MathFunction.rootMeanSquaredError(result1, dataSets.testSets.get(i), fullSet));

                System.out.println();
            }

            System.out.println("Calculating regression loss functions");
            System.out.println("Mean Absolute error is : " + df.format(absError / numTrainingSets) + "  Root Mean Squared Error : " + df.format(RMSE / numTrainingSets));
            printer.println("Mean Absolute error is : " + df.format(absError / numTrainingSets) + "  Root Mean Squared Error : " + df.format(RMSE / numTrainingSets));

        }

        // calculates and returns loss functions for classification data sets
        else {

            double precisionAvg = 0;
            double recallAvg = 0;
            double accuracyAvg = 0;

            // calculate the loss functions
            for (int i = 0; i < numTrainingSets; i++) {
                int numClusters = editedSets.get(i).size();

                // actually runs the algorithms
                System.out.println("Running K-Means on the training set " + (i + 1));
                ArrayList<ArrayList<String>> KmeansSet = Algorithms.Kmeans(dataSets.trainingSets.get(i), numClusters );
                System.out.println("Passing the condensed data set " + (i + 1) + " into KNN");
                ArrayList<String> result = Algorithms.KNN(KmeansSet,dataSets.testSets.get(i), 1, regression, euclidean);

                result = MathFunction.processConfusionMatrix(result, dataSets.testSets.get(i));
                precisionAvg += Double.parseDouble(result.get(0));
                recallAvg += Double.parseDouble(result.get(1));
                accuracyAvg += Double.parseDouble(result.get(2));

                System.out.println();
            }

            System.out.println("Calculating classification loss functions");
            System.out.println("Precision is: " + df.format(precisionAvg / numTrainingSets) + "\nRecall is: " + df.format(recallAvg / numTrainingSets) + "\nAccuracy is: " + df.format(accuracyAvg / numTrainingSets));
            printer.println("Precision is: " + df.format(precisionAvg / numTrainingSets) + "\nRecall is: " + df.format(recallAvg / numTrainingSets) + "\nAccuracy is: " + df.format(accuracyAvg / numTrainingSets));
        }
    }

    // driver for Partioning Around Medoids algorithm
    public void runKPAM(boolean regression, boolean euclidean){

        System.out.println("\n---");
        System.out.println("Performing the algorithm with 10-fold cross validation\n");

        // calculates and returns loss functions for regression data sets
        if(regression){

            System.out.println("\nCalculating regression loss functions");

            // calculate the root mean squared and absolute error
            double RMSE = 0;
            double absError = 0;
            for (int i = 0; i < numTrainingSets; i++) {

                int numClusters = dataSets.trainingSets.size() / 4; // only uses n / 4 as per instruction

                // actually runs the algorithms
                System.out.println("Running PAM on the training set " + (i + 1));
                ArrayList<ArrayList<String>> KPAMSet = Algorithms.AlternativePAM(dataSets.trainingSets.get(i), numClusters);
                System.out.println("Passing the condensed data set " + (i + 1) + " into KNN");
                ArrayList<String> result = Algorithms.KNN(KPAMSet, dataSets.testSets.get(i), 1, regression, euclidean);

                absError += Double.parseDouble(MathFunction.meanAbsoluteError(result, dataSets.testSets.get(i), fullSet));
                RMSE += Double.parseDouble(MathFunction.rootMeanSquaredError(result, dataSets.testSets.get(i), fullSet));

                System.out.println();
            }

            System.out.println("Calculating regression loss functions");
            System.out.println("Mean Absolute error is : " + df.format(absError / numTrainingSets) + "\nRoot Mean Squared Error : " + df.format(RMSE / numTrainingSets));
            printer.println("Mean Absolute error is : " + df.format(absError / numTrainingSets) + "\nRoot Mean Squared Error : " + df.format(RMSE / numTrainingSets));

        }

        // calculates and returns loss functions for classification data sets
        else {

            System.out.println("\nCalculating classification loss functions");

            double precisionAvg = 0;
            double recallAvg = 0;
            double accuracyAvg = 0;

            for (int i = 0; i < numTrainingSets; i++) {
                int numClusters = editedSets.get(i).size();

                // actually runs the algorithms
                System.out.println("Running PAM on the training set " + (i + 1));
                ArrayList<ArrayList<String>> KPAMSet = Algorithms.AlternativePAM(dataSets.trainingSets.get(i), numClusters);
                System.out.println("Passing the condensed data set " + (i + 1) + " into KNN");
                ArrayList<String> result = Algorithms.KNN(KPAMSet,dataSets.testSets.get(i), 1, regression, euclidean);

                result = MathFunction.processConfusionMatrix(result, dataSets.testSets.get(i));
                precisionAvg += Double.parseDouble(result.get(0));
                recallAvg += Double.parseDouble(result.get(1));
                accuracyAvg += Double.parseDouble(result.get(2));

                System.out.println();
            }

            System.out.println("Calculating classification loss functions");
            System.out.println("Precision is: " + df.format(precisionAvg / numTrainingSets) + "\nRecall is: " + df.format(recallAvg / numTrainingSets) + "\nAccuracy is: " + df.format(accuracyAvg / numTrainingSets));
            printer.println("Precision is: " + df.format(precisionAvg / numTrainingSets) + "\nRecall is: " + df.format(recallAvg / numTrainingSets) + "\nAccuracy is: " + df.format(accuracyAvg / numTrainingSets));
        }
    }
}
