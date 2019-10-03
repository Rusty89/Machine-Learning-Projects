import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Algorithms {


    //base KNN function
    public static ArrayList<String> KNN(ArrayList<ArrayList<String>>trainingData, ArrayList<ArrayList<String>>testingData, int k, boolean regression, boolean euclidean){
        //variables to hold sizes of things
        int lengthOfTrainingSet = trainingData.size();
        int lengthOfTestingSet = testingData.size();
        int lengthOfFeatures = trainingData.get(0).size()-2;
        int classIndex = trainingData.get(0).size()-1;
        ArrayList<String> results = new ArrayList<String>();

        for (int i = 0; i <lengthOfTestingSet; i++) {

            ArrayList<String> distanceToAllPoints = new ArrayList<String>();
            ArrayList<Integer> indicesOfMinimumDistances = new ArrayList<Integer>();
            ArrayList<String> classificationOfNeighbors = new ArrayList<String>();

            for (int j = 0; j <lengthOfTrainingSet ; j++) {
                //calculate distance between all training data points and one testing point
                List<String> trainingFeatures = trainingData.get(j).subList(0,lengthOfFeatures);
                List<String> testingFeatures = testingData.get(i).subList(0,lengthOfFeatures);
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

            //put neighbor's classification's in a list
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



    public static ArrayList<String> EditedKNN(ArrayList<ArrayList<String>> trainingData, ArrayList<ArrayList<String>> testingData, ArrayList<ArrayList<String>>validationSet, int k, boolean regression, boolean euclidean) {

        int lengthOfTrainingData= trainingData.size();
        int indexOfClassification = trainingData.get(0).size()-1;
        double currentPrecision = 0.00000001;
        double currentRecall=0.0000001;
        double currentAccuracy=.0000001;
        double prevPrecision = 0;
        double prevRecall=0;
        double prevAccuracy=0;
        //arraylist to hold the indexes of the points that were incorrectly classified
        ArrayList<Integer> badSamplePoints= new ArrayList<>();
        //holds the edited training data
        ArrayList<ArrayList<String>> editedTrainingData =(ArrayList<ArrayList<String>>) trainingData.clone();

        //while edited set still improving
        while(currentAccuracy>prevAccuracy && currentPrecision>prevPrecision && currentRecall>prevRecall){
            //run each point in the training data through KNN
            for (int i = 0; i < editedTrainingData.size(); i++) {
                ArrayList<ArrayList<String>> samplePoint = new ArrayList<>();
                samplePoint.add(editedTrainingData.get(i));
                ArrayList<String> classification = KNN(editedTrainingData,samplePoint, k, regression,euclidean);
                //sample point classified correctly
                if(classification.get(0).equals(samplePoint.get(0).get(indexOfClassification))){
                    //do nothing
                }else{
                    //remove bad data from the set
                    editedTrainingData.remove(editedTrainingData.get(i));
                }
            }
            ArrayList<String>results= new ArrayList<>();
            results= MathFunction.processConfusionMatrix(KNN(editedTrainingData,validationSet,k,regression,euclidean), validationSet);

            prevAccuracy= currentAccuracy;
            prevPrecision=currentPrecision;
            prevRecall=currentRecall;

            currentPrecision= Double.parseDouble(results.get(0));
            currentRecall = Double.parseDouble(results.get(1));
            currentAccuracy= Double.parseDouble(results.get(2));

        }





        return  KNN(editedTrainingData,testingData, k, regression, euclidean);
    }

}
