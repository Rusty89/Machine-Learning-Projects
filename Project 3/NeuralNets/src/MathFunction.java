/* A class that holds helper Math functions for our algorithms. They are statically called whenever needed in the algorithm
    implementations or when running the algorithm in the Data class methods.
 */

import java.util.*;

public class MathFunction {

    public static double logisiticActivationFunction(double input){
        double output= 1/(1-Math.exp(input));
        return output;
    }

    public static double gaussianKernelActivation(List<String> inputVector, List<String> center, double sigma) {
        double output = 0;
        output = Math.exp(-Math.pow(euclideanDistance(inputVector, center), 2) / (2 * Math.pow(sigma, 2)));
        return output;
    }


    // calculates euclidean distance defined by (x1 - x2)^2
    public static double euclideanDistance(List<String> X1Vector, List<String> X2Vector){
        double result = 0;
        int sizeOfX1andX2 = X1Vector.size();

        // iterates through the length of the passed in vector
        for (int i = 0; i < sizeOfX1andX2; i++) {

            // sqrt not used as it won't effect the result, but will cost computing time
            double x = Double.parseDouble(X1Vector.get(i));
            double y = Double.parseDouble(X2Vector.get(i));

            // summation of (x1i-x2i)^2, euclidean distance
            result += Math.pow((x - y), 2);
        }

        // prevent returning bad values
        if(Double.isInfinite(result)){
            result = Double.MAX_VALUE;
        }
        if(Double.isNaN(result)){
            result = Double.MAX_VALUE;
        }

        return result;
    }

    // calculates hamming distance between two points
    public static double hammingDistance(List<String> point1, List<String> point2){
        double result = 0;

        // iterates over the point vector
        for (int i = 0; i < point1.size(); i++) {

            boolean distanceIsZero = point1.get(i).equals(point2.get(i));

            // only increments if hamming distance is not 0 between two points
            if(!distanceIsZero){
                result++;
            }
        }

        // prevent returning bad values
        if(Double.isInfinite(result)){
            result = Double.MAX_VALUE;
        }
        if(Double.isNaN(result)){
            result = Double.MAX_VALUE;
        }
        return result;
    }

    // calculates the average of inputted data points
    public static String average(ArrayList<String> input) {
        double sum = 0;

        // sums over all data points
        for (int i = 0; i < input.size(); i++) {
            sum += Double.parseDouble(input.get(i));
        }
        sum /= input.size(); // divides by number of data points to get mean

        return String.valueOf(sum);
    }

    // calculates the mode: the value which occurs the most
    public static String mode(ArrayList<String> input) {
        String mode = "";
        int maxIndex = 0;
        int currMax = 0;

        // iterates over each data point and determines value
        for (int i = 0; i < input.size(); i++) {

            int countOfClass = Collections.frequency(input, input.get(i));
            if(countOfClass >= currMax){
                maxIndex = i;
                currMax = countOfClass;
            }
        }
        mode = input.get(maxIndex);

        return mode;
    }

    // method to use a confusion matrix to calculate our loss functions
    public static ArrayList<String> processConfusionMatrix(ArrayList<String> results, ArrayList<ArrayList<String>> testData) {

        int lengthOfData = results.size();
        int classIndex = testData.get(0).size() - 1;

        // initialize classifications in the confusion matrix as the first row
        ArrayList<ArrayList<String>> confusionMatrix= new ArrayList<>();
        confusionMatrix.add(new ArrayList<>());

        // generating a list of all unique classes in the training and test sets
        for (int i = 0; i < lengthOfData ; i++) {
            if(!confusionMatrix.get(0).contains(testData.get(i).get(classIndex))) {
                confusionMatrix.get(0).add(testData.get(i).get(classIndex));
            }
            if(!confusionMatrix.get(0).contains(results.get(i))){
                confusionMatrix.get(0).add(results.get(i));
            }
        }

        // initialize the rest of the matrix as an n by n matrix below the classification row initialized with 0
        for (int i = 0; i < confusionMatrix.get(0).size(); i++) {
            confusionMatrix.add(new ArrayList<>());
            for (int j = 0; j < confusionMatrix.get(0).size(); j++) {
                confusionMatrix.get(i + 1).add("0");
            }
        }

        // populate the matrix by indexing guesses by their true value and the guess
        for (int i = 0; i < lengthOfData; i++) {
            String guess = results.get(i);
            String actual = testData.get(i).get(testData.get(0).size() - 1);
            int indexOfActual = confusionMatrix.get(0).indexOf(actual);
            int indexOfGuess = confusionMatrix.get(0).indexOf(guess);
            int currentValAtPos = Integer.parseInt(confusionMatrix.get(indexOfActual + 1).get(indexOfGuess));
            currentValAtPos++;
            confusionMatrix.get(indexOfActual+1).set(indexOfGuess, String.valueOf(currentValAtPos));
        }

        // calculate truePos, falsePos, falseNeg, and totalPos by indexing the confusion matrix
        double truePos = 0;
        double totalPos = 0;
        double falsePos = 0;
        double falseNeg =  0;
        double precisionSum = 0;
        double recallSum = 0;

        // iterating through the confusion matrix 
        for (int i = 1; i <confusionMatrix.size(); i++) {
            // grabbing all positive values from the diagonal
            totalPos += Integer.parseInt(confusionMatrix.get(i).get(i - 1));
            // grabbing the current true positive from this point for class i
            truePos = Integer.parseInt(confusionMatrix.get(i).get(i - 1));
            for (int j = 0; j < confusionMatrix.get(0).size(); j++) {
                // false negatives grabbed by row
                falseNeg += Integer.parseInt(confusionMatrix.get(i).get(j));
                // false positives are grabbed from column
                falsePos += Integer.parseInt(confusionMatrix.get(j + 1).get(i - 1));
            }

            // get rid of true pos value in these results
            falseNeg -= truePos;
            falsePos -= truePos;

            // calculates the precision and recall for each class
            // adds it to the summation to later be divided by
            // the number of classes

            // protection against division by zero errors
            if(!Double.isNaN(truePos / truePos + falsePos)){
                precisionSum += truePos / (truePos  +  falsePos);
            }
            if(!Double.isNaN(truePos / truePos + falseNeg)){
                recallSum += truePos / (truePos + falseNeg);
            }
        }

        // divides sums by number of classes to get overall precision and recall
        precisionSum /= confusionMatrix.get(0).size();
        recallSum /= confusionMatrix.get(0).size();

        // calculate overall accuracy by dividing all truePos results and dividing by all guessed data points
        double accuracy = totalPos/results.size();
        ArrayList<String> result = new ArrayList<String>();
        result.add(String.valueOf(precisionSum));
        result.add(String.valueOf(recallSum));
        result.add(String.valueOf(accuracy));
        return result;
    }

    // calculate the regression loss function for root mean squared error
    public static String rootMeanSquaredError(ArrayList<String> results, ArrayList<ArrayList<String>> testData,  ArrayList<ArrayList<String>> fullSet) {
        double sum = 0;
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;

        // get max and min values in the set to normalize results
        for (int i = 0; i < fullSet.size(); i++) {
            if(max <= Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size() - 1))){
                max = Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size() - 1));
            }
            if(min >= Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size() - 1))){
                min = Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size() - 1));
            }
        }

        // find the guess and actual value for each data point
        for (int i = 0; i < testData.size(); i++) {
            double guess = Double.parseDouble(results.get(i));
            double actual = Double.parseDouble(testData.get(i).get(testData.get(0).size() - 1));

            sum += Math.pow((guess - actual), 2);
        }

        // average the summation
        sum /= testData.size();
        sum = Math.sqrt(sum);

        // normalize to a percentage
        sum /= (max - min);
        return String.valueOf(sum);
    }

    // calculates the regression loss function for mean absolute error
    public static String meanAbsoluteError(ArrayList<String> results, ArrayList<ArrayList<String>> testData,ArrayList<ArrayList<String>> fullSet){
        double sum = 0;
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;

        // get max and min values in the set to normalize results
        for (int i = 0; i < fullSet.size(); i++) {
            if(max <= Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size() - 1))){
                max = Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size() - 1));
            }
            if(min >= Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size() - 1))){
                min = Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size() - 1));
            }
        }

        // calculates guess and actual result for each data point
        for (int i = 0; i < testData.size(); i++) {
            double guess = Double.parseDouble(results.get(i));
            double actual = Double.parseDouble(testData.get(i).get(testData.get(0).size() - 1));
            sum +=  Math.abs(guess - actual);
        }

        // average the summation
        sum /= testData.size();

        // normalize to a percentage
        sum /= (max - min);
        return String.valueOf((sum));
    }

    // finds and returns a random centroid for K-Means
    public static ArrayList<String> randomCentroid(int numFeatures){
        ArrayList<String> centroid = new ArrayList<>();

        // picks a random centroid for each feature
        for (int i = 0; i < numFeatures; i++){
            centroid.add(Double.toString(Math.random()));
        }
        return centroid;
    }

    // distortion calculation used in the traditional PAM algorithm for determining distances from the medoid
    public static double distortion(ArrayList<ArrayList<String>> trainingData, ArrayList<ArrayList<String>> clusterMedoids, boolean euclidean){
        int numMedoids = clusterMedoids.size();
        int lengthOfFeatures = trainingData.get(0).size() - 2;
        double distortion = 0;
        for (int i = 0; i <trainingData.size(); i++) {
            ArrayList<Double> distanceToAllPoints= new ArrayList<>();
            for (int j = 0; j < numMedoids; j++) {

                // calculate distance between all training data points and a medoid
                List<String> trainingFeatures = trainingData.get(i).subList(0, lengthOfFeatures);
                List<String> medoidFeatures = clusterMedoids.get(j).subList(0, lengthOfFeatures);

                // uses euclidean or hamming distance as appropriate for  the data
                if(euclidean){
                    distanceToAllPoints.add(MathFunction.euclideanDistance(trainingFeatures, medoidFeatures));
                }else{
                    distanceToAllPoints.add(MathFunction.hammingDistance(trainingFeatures, medoidFeatures));
                }
            }
            // add min distance to distortion total
            distortion += Collections.min(distanceToAllPoints);
        }
        return distortion;
    }
}
