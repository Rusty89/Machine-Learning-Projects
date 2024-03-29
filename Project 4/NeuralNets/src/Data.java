/* The Data Class is a parent to inheriting classes that are defined for each data set. It handles reading in files, pre-processing
    normalizing data, bucketizing for 10-fold cross validation, and methods for running each of the KNNAlgorithms.
 */

import java.io.File;
import java.util.*;

public class Data {
    public int numClasses;
    public ArrayList<ArrayList<String>> fullSet = new ArrayList<>();
    public CVS dataSets = new CVS();
    private final int numTrainingSets = 10; // defines training sets for 10-fold cross validation
    public int numClassifications = 0;

    // method to read in our data sets and convert them to an Java ArrayList for parsing
    public void fileTo2dStringArrayList(File inputFile) throws Exception {

        final int maxExamplesToRun = 2000; // max number of lines of data, to keep test manageable

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

    // sets the number of different classifications that can exist in categorical data sets
    public void findNumClassifications() {
        ArrayList<String>possibleOutcomes= new ArrayList<>();
        for (int i = 0; i < fullSet.size(); i++) {
            possibleOutcomes.add(fullSet.get(i).get(fullSet.get(0).size() - 1));
        }

        Set<String> set = new HashSet<String>(possibleOutcomes);
        this.numClassifications = set.size();

    }
}
