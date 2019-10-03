import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Algorithms {



    public static ArrayList<String> KNN(ArrayList<ArrayList<String>> trainingData,ArrayList<ArrayList<String>> validationData,ArrayList<ArrayList<String>> testingData, int k, boolean regression, boolean euclidean){
        int lengthOfTrainingSet = trainingData.size();
        int lengthOfTestingSet = testingData.size();
        int lengthOfFeatures = trainingData.get(0).size()-1;
        int classIndex = trainingData.get(0).size()-1;
        ArrayList<String> results = new ArrayList<String>();

        for (int i = 0; i <lengthOfTestingSet; i++) {

            ArrayList<String> distanceToAllPoints = new ArrayList<String>();
            ArrayList<Integer> indicesOfMinimumDistances = new ArrayList<Integer>();
            ArrayList<String> classificationOfNeighbors = new ArrayList<String>();

            for (int j = 0; j <lengthOfTrainingSet ; j++) {
                //calculate distance between all training data points and one testing point
                List<String> trainingFeatures = trainingData.get(j).subList(0,lengthOfFeatures);
                List<String> testingFeatures = trainingData.get(i).subList(0,lengthOfFeatures);
                //uses euclidean or hamming distance as appropriate for  the data
                if(euclidean){
                    distanceToAllPoints.add(MathFunction.euclideanDistance(trainingFeatures, testingFeatures));
                }else{
                    distanceToAllPoints.add(MathFunction.hammingDistance(trainingFeatures, testingFeatures));
                }

            }

            //find indices of the k nearest neighbors
            for (int j = 0; j <k ; j++) {
                //grab first minimum value in distance and record that index
                int indexOfMinValue= distanceToAllPoints.indexOf(Collections.min(distanceToAllPoints));
                indicesOfMinimumDistances.add(indexOfMinValue);

                //set that index to max value so it will not be used twice
                distanceToAllPoints.set(indexOfMinValue, Double.MAX_VALUE+"");
            }

            //put neighbors classifications in a list
            for (int j = 0; j < k ; j++) {
                String classification = trainingData.get(indicesOfMinimumDistances.get(j)).get(classIndex);
                classificationOfNeighbors.add(classification);
            }


            //find average if using regression data
            if(regression){
                results.add(MathFunction.average(classificationOfNeighbors));
            //find mode with classifcation data
            }else{
                results.add(MathFunction.mode(classificationOfNeighbors));
            }

        }
        //returns the list of guessed results for the test set
        return results;
    }
}
