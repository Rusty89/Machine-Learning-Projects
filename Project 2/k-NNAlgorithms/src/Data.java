import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

public class Data {
    private static PrintWriter printer;
    public ArrayList<ArrayList<String>> fullSet = new ArrayList<>();
    public CVS dataSets = new CVS();
    private final int numTrainingSets = 10;
    private final int kValueSelections[] = {1, 3, 5}; // choose odd values for k to avoid tie-breakers
    private ArrayList<ArrayList<ArrayList<String>>> editedSets = new ArrayList<>();
    private ArrayList<ArrayList<ArrayList<String>>> condensedSets = new ArrayList<>();

    public void runTests(boolean regression, boolean euclidean, int numClusters, String dataName) throws IOException {

        FileWriter filer = new FileWriter(dataName + " results.txt");
        printer = new PrintWriter(filer);
        System.out.println("Begin KNN test");
        printer.println("Begin KNN test");
        runKNN(regression,euclidean);
        System.out.println("End test\n\n");
        printer.println("End test\n\n");
        if(!regression){
            System.out.println("Begin Condensed KNN test");
            printer.println("Begin Condensed KNN test");
            runCondensedKNN(euclidean);
            System.out.println("End test\n\n");
            printer.println("End test\n\n");
            System.out.println("Begin Edited KNN test");
            printer.println("Begin Edited KNN test");
            runEditedKNN(euclidean);
            System.out.println("End test\n\n");
            printer.println("End test\n\n");
        }
        System.out.println("Begin KMeans test");
        printer.println("Begin KMeans test");
        runKMeans(regression,euclidean, numClusters);
        System.out.println("End test\n\n");
        printer.println("End test\n\n");
        System.out.println("Begin K Medoids PAM test");
        printer.println("Begin K Medoids PAM test");
        runKPAM(regression,euclidean, numClusters);
        System.out.println("End test");
        printer.println("End test");
        printer.close();
        filer.close();
    }

    public void fileTo2dStringArrayList(File inputFile) throws Exception{

        Scanner sc = new Scanner(inputFile); // read in input file as an array list
        int maxCount = 400; // max number of lines of data, to keep test manageable

        while (sc.hasNextLine()){
            ArrayList<String> line= new ArrayList<>(Arrays.asList(sc.nextLine().split(",")));;
            fullSet.add(line);
        }

        // shuffle the input data into a random order
        Collections.shuffle(fullSet);

        // remove items to make dataset manageable
        while(fullSet.size() > maxCount){
            fullSet.remove(0);
        }

    }

    public void normalizeData(){

        ArrayList<Double> max = new ArrayList<Double>();
        ArrayList<Double> min = new ArrayList<Double>();
        int indexOfLastTrait = fullSet.get(0).size() - 1;
        int sizeOfSet = fullSet.size();

        // initialize mins and maxes
        for (int i = 0; i < indexOfLastTrait ; i++) {
            max.add(Double.MIN_VALUE);
            min.add(Double.MAX_VALUE);
        }
        // checks each column of the dataset
        for (int j = 0; j < indexOfLastTrait; j++) {
            for (int i = 0; i < sizeOfSet ; i++) {

                // find max and min values in each column of dataset
                max.set(j, Double.max(max.get(j), Double.parseDouble(fullSet.get(i).get(j))));
                min.set(j, Double.min(min.get(j), Double.parseDouble(fullSet.get(i).get(j))));
            }
        }

        // go through the dataset and normalize values between 0-1
        for (int i = 0; i < sizeOfSet; i++) {
            for (int j = 0; j < indexOfLastTrait; j++) {
                // if max and min are the same, normalize it to a 1
                if((max.get(j) - min.get(j)) == 0){
                    fullSet.get(i).set(j, "1.0");
                }else{

                    // normalize the value using equation (x-min)/(max-min)
                    double normalizedValue = (Double.parseDouble(fullSet.get(i).get(j))-min.get(j))/(max.get(j)-min.get(j));
                    fullSet.get(i).set(j, normalizedValue + "");
                }
            }
        }
    }

    public void bucketize() {

        int countTrainingSet = 0;
        double eightyPercentOfData = 0.8 * fullSet.size();
        double twentyPercentOfData = 0.2 * fullSet.size();
        double tenPercentOfData = 0.1 * fullSet.size();
        for (int i = 0; i < numTrainingSets; i++) {

            // initializes new ArrayLists to store sets in the CVS structure
            dataSets.trainingSets.add(new ArrayList<ArrayList<String>>());
            dataSets.validationSets.add(new ArrayList<ArrayList<String>>());
            dataSets.testSets.add(new ArrayList<ArrayList<String>>());

            // generates a training set with 80% of the data
            for (int j = 0; j < eightyPercentOfData; j++) {
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
            int countValidationAndTest = countTrainingSet;
            for (int j = 0; j < twentyPercentOfData ; j++) {

                // generates validation set with next 10% of data
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




    public void runKNN(boolean regression, boolean euclidean){

        if(regression){
            for (int k : kValueSelections) {
                double RMSE = 0; // root mean squared error
                double absError = 0;
                for (int i = 0; i < numTrainingSets; i++) {
                    ArrayList<String> result1 = Algorithms.KNN(dataSets.trainingSets.get(i), dataSets.testSets.get(i), k, regression, euclidean);
                    absError += Double.parseDouble(MathFunction.meanAbsoluteError(result1, dataSets.testSets.get(i), fullSet));
                    RMSE += Double.parseDouble(MathFunction.rootMeanSquaredError(result1, dataSets.testSets.get(i), fullSet));
                }

                System.out.println("Mean Absolute error is : " + absError / 10 + "  Root Mean Squared Error : " + RMSE / 10);
                printer.println("Mean Absolute error is : " + absError / 10 + "  Root Mean Squared Error : " + RMSE / 10);
            }
        }else{
            for (int k : kValueSelections) {
                double precisionAvg = 0;
                double recallAvg = 0;
                double accuracyAvg = 0;

                for (int i = 0; i < numTrainingSets; i++) {
                    ArrayList<String>result=Algorithms.KNN(dataSets.trainingSets.get(i),dataSets.testSets.get(i), k, regression, euclidean);
                    result = MathFunction.processConfusionMatrix(result, dataSets.testSets.get(i));
                    precisionAvg += Double.parseDouble(result.get(0));
                    recallAvg += Double.parseDouble(result.get(1));
                    accuracyAvg += Double.parseDouble(result.get(2));

                }
                System.out.println("Precision is: " + precisionAvg / 10 + " Recall is:" + recallAvg / 10 + " Accuracy is: " + accuracyAvg / 10);
                printer.println("Precision is: " + precisionAvg / 10 + " Recall is:" + recallAvg / 10 + " Accuracy is: " + accuracyAvg / 10);
            }
        }

    }


    public void runEditedKNN(boolean euclidean){

        for (int k : kValueSelections) {
            double precisionAvg = 0;
            double recallAvg = 0;
            double accuracyAvg = 0;

            for (int i = 0; i < numTrainingSets; i++) {
                ArrayList<ArrayList<String>> editedSet = Algorithms.EditedKNN(dataSets.trainingSets.get(i), dataSets.validationSets.get(i), k, false, true);
                ArrayList<String>result=Algorithms.KNN(editedSet,dataSets.testSets.get(i), k, false, euclidean);
                result = MathFunction.processConfusionMatrix(result, dataSets.testSets.get(i));
                precisionAvg += Double.parseDouble(result.get(0));
                recallAvg += Double.parseDouble(result.get(1));
                accuracyAvg += Double.parseDouble(result.get(2));


                editedSets.add(editedSet);

            }
            System.out.println("Precision is: " + precisionAvg / 10 + " Recall is:" + recallAvg / 10 + " Accuracy is: " + accuracyAvg / 10);
            printer.println("Precision is: " + precisionAvg / 10 + " Recall is:" + recallAvg / 10 + " Accuracy is: " + accuracyAvg / 10);
        }
    }

    public void runCondensedKNN( boolean euclidean){
        for (int k : kValueSelections) {

            double precisionAvg = 0;
            double recallAvg = 0;
            double accuracyAvg = 0;

            for (int i = 0; i < numTrainingSets; i++) {
                ArrayList<ArrayList<String>> condensedSet = Algorithms.CondensedKNN(dataSets.trainingSets.get(i), euclidean);
                ArrayList<String>result1=Algorithms.KNN(condensedSet,dataSets.testSets.get(i), k, false, euclidean);
                result1 = MathFunction.processConfusionMatrix(result1, dataSets.testSets.get(i));
                precisionAvg += Double.parseDouble(result1.get(0));
                recallAvg += Double.parseDouble(result1.get(1));
                accuracyAvg += Double.parseDouble(result1.get(2));
                condensedSets.add(condensedSet);
            }
            System.out.println("Precision is: " + precisionAvg / 10 + " Recall is:" + recallAvg / 10 + " Accuracy is: " + accuracyAvg / 10);
            printer.println("Precision is: " + precisionAvg / 10 + " Recall is:" + recallAvg / 10 + " Accuracy is: " + accuracyAvg / 10);
        }
    }

    public void runKMeans(boolean regression, boolean euclidean, int numClusters){
        if(regression){

            double RMSE = 0; // root mean squared error
            double absError = 0;
            for (int i = 0; i < numTrainingSets; i++) {
                ArrayList<ArrayList<String>> KmeansSet = Algorithms.Kmeans(dataSets.trainingSets.get(i), numClusters );
                ArrayList<String> result1 = Algorithms.KNN(KmeansSet, dataSets.testSets.get(i), 1, regression, euclidean);
                absError += Double.parseDouble(MathFunction.meanAbsoluteError(result1, dataSets.testSets.get(i), fullSet));
                RMSE += Double.parseDouble(MathFunction.rootMeanSquaredError(result1, dataSets.testSets.get(i), fullSet));
            }
            System.out.println("Mean Absolute error is : " + absError / 10 + "  Root Mean Squared Error : " + RMSE / 10);
            printer.println("Mean Absolute error is : " + absError / 10 + "  Root Mean Squared Error : " + RMSE / 10);

        }else{

            double precisionAvg = 0;
            double recallAvg = 0;
            double accuracyAvg = 0;

            for (int i = 0; i < numTrainingSets; i++) {
                ArrayList<ArrayList<String>> KmeansSet = Algorithms.Kmeans(condensedSets.get(i), numClusters );
                ArrayList<String>result1 = Algorithms.KNN(KmeansSet,dataSets.testSets.get(i), 1, regression, euclidean);
                result1 = MathFunction.processConfusionMatrix(result1, dataSets.testSets.get(i));
                precisionAvg += Double.parseDouble(result1.get(0));
                recallAvg += Double.parseDouble(result1.get(1));
                accuracyAvg += Double.parseDouble(result1.get(2));
            }
            System.out.println("Precision is: " + precisionAvg / 10 + " Recall is:" + recallAvg / 10 + " Accuracy is: " + accuracyAvg / 10);
            printer.println("Precision is: " + precisionAvg / 10 + " Recall is:" + recallAvg / 10 + " Accuracy is: " + accuracyAvg / 10);
        }


    }

    public void runKPAM(boolean regression, boolean euclidean, int numClusters){
        if(regression){

            double RMSE = 0; // root mean squared error
            double absError = 0;
            for (int i = 0; i < numTrainingSets; i++) {
                ArrayList<ArrayList<String>> KPAMSet = Algorithms.PAM(dataSets.trainingSets.get(i), numClusters );
                ArrayList<String> result1 = Algorithms.KNN(KPAMSet, dataSets.testSets.get(i), 1, regression, euclidean);
                absError += Double.parseDouble(MathFunction.meanAbsoluteError(result1, dataSets.testSets.get(i), fullSet));
                RMSE += Double.parseDouble(MathFunction.rootMeanSquaredError(result1, dataSets.testSets.get(i), fullSet));
            }
            System.out.println("Mean Absolute error is : " + absError / 10 + "  Root Mean Squared Error : " + RMSE / 10);
            printer.println("Mean Absolute error is : " + absError / 10 + "  Root Mean Squared Error : " + RMSE / 10);

        }else{

            double precisionAvg = 0;
            double recallAvg = 0;
            double accuracyAvg = 0;

            for (int i = 0; i < numTrainingSets; i++) {
                ArrayList<ArrayList<String>> KPAMSet = Algorithms.PAM(condensedSets.get(i), numClusters );
                ArrayList<String>result1=Algorithms.KNN(KPAMSet,dataSets.testSets.get(i), 1, regression, euclidean);
                result1 = MathFunction.processConfusionMatrix(result1, dataSets.testSets.get(i));
                precisionAvg += Double.parseDouble(result1.get(0));
                recallAvg += Double.parseDouble(result1.get(1));
                accuracyAvg += Double.parseDouble(result1.get(2));
            }
            System.out.println("Precision is: " + precisionAvg / 10 + " Recall is:" + recallAvg / 10 + " Accuracy is: " + accuracyAvg / 10);
            printer.println("Precision is: " + precisionAvg / 10 + " Recall is:" + recallAvg / 10 + " Accuracy is: " + accuracyAvg / 10);
        }


    }

}
