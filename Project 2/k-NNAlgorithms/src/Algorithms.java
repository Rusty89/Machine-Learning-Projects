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

            ArrayList<Double> distanceToAllPoints = new ArrayList<Double>();
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
                distanceToAllPoints.set(indexOfMinValue, Double.MAX_VALUE);
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

        ArrayList<Integer> badSamplePoints= new ArrayList<>();//arraylist to hold the indexes of the points that were incorrectly classified

        ArrayList<ArrayList<String>> editedTrainingData =(ArrayList<ArrayList<String>>) trainingData.clone();//holds the edited training data
        ArrayList<ArrayList<String>> finalEditedTrainingData =new ArrayList<>();


        while(currentAccuracy>prevAccuracy || currentPrecision>prevPrecision || currentRecall>prevRecall){//while edited set is still improving

            finalEditedTrainingData=(ArrayList<ArrayList<String>>) editedTrainingData.clone();//copy the editedData into a potential final set
            for (int i = 0; i < editedTrainingData.size(); i++) {//go through each point in the editedData
                ArrayList<ArrayList<String>> samplePoint = new ArrayList<>();//make a new sample point
                samplePoint.add(editedTrainingData.get(i));//give it the value of editedTraining data at i
                ArrayList<String> classification = KNN(editedTrainingData,samplePoint, k, regression,euclidean);

                if(classification.get(0).equals(samplePoint.get(0).get(indexOfClassification))){ //sample point classified correctly
                    //do nothing
                }else{//sample point not correctly classified
                    editedTrainingData.remove(editedTrainingData.get(i));//remove bad data from the set
                }
            }

            //run the editedTraining set with the validation set to determine if
            //edited increased or decreased accuracy, precision or recall
            ArrayList<String> results= MathFunction.processConfusionMatrix(KNN(editedTrainingData,validationSet,k,regression,euclidean), validationSet);

            //set prevs and currs to determine if a decrease in accuracy,precision or recall has
            //occurred, exit while loop if so.
            prevAccuracy= currentAccuracy;
            prevPrecision=currentPrecision;
            prevRecall=currentRecall;

            currentPrecision= Double.parseDouble(results.get(0));
            currentRecall = Double.parseDouble(results.get(1));
            currentAccuracy= Double.parseDouble(results.get(2));

        }
        //return the set of edited training data just prior to run that had a decrease
        //in precision, accuracy or recall
        return  KNN(finalEditedTrainingData,testingData, k, regression, euclidean);
    }



    public static ArrayList<String> CondensedKNN(ArrayList<ArrayList<String>>trainingData, ArrayList<ArrayList<String>>testingData, int k, boolean regression, boolean euclidean){
        //initialize empty set
        ArrayList<ArrayList<String>> condensedTrainingData=new ArrayList<>();
        int currSizeOfCondensed=-1;
        int indexOfClassification=trainingData.get(0).size()-1;
        int lengthOfFeatures = trainingData.get(0).size()-2;
        //while  condensedTrainingData size is increasing
        while(condensedTrainingData.size()>currSizeOfCondensed){
            //randomize order
            currSizeOfCondensed=condensedTrainingData.size();
            Collections.shuffle(trainingData);
            for (int i = 0; i <trainingData.size() ; i++) {
                //if condensed has no elements, add the first
                if(condensedTrainingData.size()==0){
                    condensedTrainingData.add(trainingData.get(i));
                }else{
                    //find minimum distance between current point and points in the condensed set
                    ArrayList<Double> distanceToAllPoints = new ArrayList<Double>();
                    for (int j = 0; j <condensedTrainingData.size() ; j++) {

                        List<String> condensedTrainingFeatures = condensedTrainingData.get(j).subList(0,lengthOfFeatures);
                        List<String> trainingFeatures = trainingData.get(i).subList(0,lengthOfFeatures);
                        if(euclidean){
                            distanceToAllPoints.add(MathFunction.euclideanDistance(condensedTrainingFeatures, trainingFeatures));
                        }else{
                            distanceToAllPoints.add(MathFunction.hammingDistance(condensedTrainingFeatures, trainingFeatures));
                        }

                    }
                    int indexOfMinValue= distanceToAllPoints.indexOf(Collections.min(distanceToAllPoints));
                    String classOfMinCondensedPoint = condensedTrainingData.get(indexOfMinValue).get(indexOfClassification);

                    //add the point to condensed training data if it's classification doesn't equal the classification of the point that
                    //is min distance from it, and it doesn't already exist in condensed
                    if(condensedTrainingData.get(indexOfMinValue).get(indexOfClassification).equals(trainingData.get(i).get(indexOfClassification))){
                        //do not add
                    }else{
                        if(condensedTrainingData.contains(trainingData.get(i))){
                            //point is already in condensed, do not add
                        }else{
                            condensedTrainingData.add(trainingData.get(i));
                        }
                    }
                }
            }
        }
        return KNN(condensedTrainingData,testingData,k,regression,euclidean);
    }

}
