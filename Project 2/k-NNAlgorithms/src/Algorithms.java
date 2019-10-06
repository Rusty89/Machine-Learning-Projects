import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.lang.Math;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class Algorithms {

    // base K-Nearest Neighbor algorithm
    public static ArrayList<String> KNN(ArrayList<ArrayList<String>>trainingData, ArrayList<ArrayList<String>>testingData, int k, boolean regression, boolean euclidean){

        // variables to hold sizes of things
        int lengthOfTrainingSet = trainingData.size();
        int lengthOfTestingSet = testingData.size();
        int lengthOfFeatures = trainingData.get(0).size() - 2;
        int classIndex = trainingData.get(0).size() - 1;
        ArrayList<String> results = new ArrayList<String>();

        for (int i = 0; i < lengthOfTestingSet; i++) {

            ArrayList<Double> distanceToAllPoints = new ArrayList<Double>();
            ArrayList<Integer> indexesOfMinimumDistances = new ArrayList<Integer>();
            ArrayList<String> classificationOfNeighbors = new ArrayList<String>();

            for (int j = 0; j < lengthOfTrainingSet; j++) {
                
                // calculate distance between all training data points and one testing point
                List<String> trainingFeatures = trainingData.get(j).subList(0, lengthOfFeatures);
                List<String> testingFeatures = testingData.get(i).subList(0, lengthOfFeatures);
               
                //uses euclidean or hamming distance as appropriate for  the data
                if(euclidean){
                    distanceToAllPoints.add(MathFunction.euclideanDistance(trainingFeatures, testingFeatures));
                }else{
                    distanceToAllPoints.add(MathFunction.hammingDistance(trainingFeatures, testingFeatures));
                }
            }

            // find indexes of the k nearest neighbors
            for (int j = 0; j < k; j++) {
                
                // grab first minimum value in distance and record that index
                int indexOfMinValue = distanceToAllPoints.indexOf(Collections.min(distanceToAllPoints));
                indexesOfMinimumDistances.add(indexOfMinValue);

                // set that index to max value so it will not be used twice
                distanceToAllPoints.set(indexOfMinValue, Double.MAX_VALUE);
            }

            // put neighbor's classification's in a list
            for (int j = 0; j < k; j++) {
                String classification = trainingData.get(indexesOfMinimumDistances.get(j)).get(classIndex);
                classificationOfNeighbors.add(classification);
            }


            // find average if using regression data
            if(regression) {
                results.add(MathFunction.average(classificationOfNeighbors));

            //find mode with classifcation data
            } else {
                results.add(MathFunction.mode(classificationOfNeighbors));
            }

        }
        // returns the list of guessed results for the test set
        return results;
    }



    public static ArrayList<String> EditedKNN(ArrayList<ArrayList<String>> trainingData, ArrayList<ArrayList<String>> testingData, ArrayList<ArrayList<String>>validationSet, int k, boolean regression, boolean euclidean) {

        int indexOfClassification = trainingData.get(0).size() - 1;
        double currentPrecision = 0;
        double currentRecall = 0;
        double currentAccuracy = 0;
        double prevPrecision = 0;
        double prevRecall = 0;
        double prevAccuracy = 0;

        ArrayList<ArrayList<String>> editedTrainingData = (ArrayList<ArrayList<String>>) trainingData.clone();
        ArrayList<ArrayList<String>> finalEditedTrainingData = (ArrayList<ArrayList<String>>) trainingData.clone();

        boolean improvementOccurred = true;

        // runs until no significant improvement in the edited set is measured
        while(improvementOccurred){

            finalEditedTrainingData = (ArrayList<ArrayList<String>>) editedTrainingData.clone();

            // iterate through each point in the edited data set
            for (int i = 0; i < editedTrainingData.size(); i++) {

                ArrayList<ArrayList<String>> samplePoint = new ArrayList<>();
                samplePoint.add(editedTrainingData.get(i));
                editedTrainingData.remove(editedTrainingData.get(i));

                ArrayList<String> classification = KNN(editedTrainingData,samplePoint, k, regression,euclidean);

                // if the sample point was classified correctly, add the point back to the dataset
                if(classification.get(0).equals(samplePoint.get(0).get(indexOfClassification))) {
                    editedTrainingData.add(i,samplePoint.get(0));
                }
            }

            // run the editedTraining set with the validation set
            // to determine if edited increased or decreased accuracy, precision or recall
            ArrayList<String> results = MathFunction.processConfusionMatrix
                    (KNN(editedTrainingData, validationSet, k, regression, euclidean), validationSet);

            // determine if a decrease in accuracy, precision, or recall has occured
            prevAccuracy = currentAccuracy;
            prevPrecision =  currentPrecision;
            prevRecall = currentRecall;

            currentPrecision = Double.parseDouble(results.get(0));
            currentRecall = Double.parseDouble(results.get(1));
            currentAccuracy = Double.parseDouble(results.get(2));

            // check if an improvement occured in all of the three categories.
            improvementOccurred = (currentAccuracy > prevAccuracy && currentPrecision > prevPrecision && currentRecall > prevRecall);

        }
        // return the set of edited training data just prior to run that saw a decrease in any metric
        return KNN(finalEditedTrainingData, testingData, k, regression, euclidean);
    }

    public static ArrayList<String> CondensedKNN(ArrayList<ArrayList<String>>trainingData, ArrayList<ArrayList<String>>testingData, int k, boolean regression, boolean euclidean) {

        //initialize empty set
        ArrayList<ArrayList<String>> condensedTrainingData=new ArrayList<>();
        int currSizeOfCondensed = -1;
        int indexOfClassification = trainingData.get(0).size() - 1;
        int lengthOfFeatures = trainingData.get(0).size() - 2;

        // loops as long as the size of the condensed datset is increasing
        while(condensedTrainingData.size() > currSizeOfCondensed) {

            // randomize order of the dataset
            currSizeOfCondensed=condensedTrainingData.size();
            Collections.shuffle(trainingData);

            // iterate through each point in the training dataset
            for (int i = 0; i < trainingData.size(); i++) {

                // add the first point if the set is empty
                if(condensedTrainingData.size() == 0) {
                    condensedTrainingData.add(trainingData.get(i));
                } else {

                    // find the minimum distance between current point and points in the condensed set
                    ArrayList<Double> distanceToAllPoints = new ArrayList<Double>();
                    for (int j = 0; j < condensedTrainingData.size(); j++) {

                        List<String> condensedTrainingFeatures = condensedTrainingData.get(j).subList(0, lengthOfFeatures);
                        List<String> trainingFeatures = trainingData.get(i).subList(0, lengthOfFeatures);

                        // use the euclidean distance formula
                        if(euclidean) {
                            distanceToAllPoints.add(MathFunction.euclideanDistance(condensedTrainingFeatures, trainingFeatures));
                        }

                        // use the hamming distance formula
                        else {
                            distanceToAllPoints.add(MathFunction.hammingDistance(condensedTrainingFeatures, trainingFeatures));
                        }
                    }

                    int indexOfMinValue= distanceToAllPoints.indexOf(Collections.min(distanceToAllPoints));
                    String classOfMinCondensedPoint = condensedTrainingData.get(indexOfMinValue).get(indexOfClassification);

                    // add the point to condensed training data if it's classification doesn't equal the classification
                    // of the point that is min distance from it, and it doesn't already exist in condensed
                    boolean classificationsAreEqual =
                            condensedTrainingData.get(indexOfMinValue).get(indexOfClassification).equals(trainingData.get(i).get(indexOfClassification));

                    if(!classificationsAreEqual) {
                        if (!condensedTrainingData.contains(trainingData.get(i))) {
                            // add point since it doesn't belong to condensed dataset yet
                            condensedTrainingData.add(trainingData.get(i));
                        }
                    }
                }
            }
        }
        return KNN(condensedTrainingData, testingData, k, regression, euclidean);
    }

    public static ArrayList<String> Kmeans (ArrayList<ArrayList<String>>trainingData,
                                            ArrayList<ArrayList<String>>testingData, int k, boolean regression,
                                            boolean euclidean, int numClasses){

        // Variables
        ArrayList<ArrayList<String>> clusterCentroids = new ArrayList<>();
        int numFeatures= trainingData.get(0).size() - 1;

        // 1) initialize cluster centroids randomly
        for (int i = 0; i < numClasses; i++){ // Number of cluster centroids = number of classes
            ArrayList<String> centroid = new ArrayList<>();
            for (int j = 0; j < numFeatures; j++){
                centroid.add(Double.toString(Math.random()));
            }
            centroid.add(Integer.toString(i));
            clusterCentroids.add(centroid);
        }

        int preventEndless = 0; // Remove for final code
        while (preventEndless < 50000) { // convert to while (true)

            ArrayList<ArrayList<String>> compareSet = new ArrayList<>();
            ArrayList<ArrayList<ArrayList<String>>> clusters = new ArrayList<>();

            for (int i = 0; i < numClasses; i++){
                clusters.add(new ArrayList<>());
            }

            // 2) Compare example distances to centroids and assign to appropriate clusters
            for (ArrayList<String> example: trainingData
                 ) {

                List<String> trainingFeatures = example.subList(0, numFeatures);
                double minDist = Double.MAX_VALUE;
                int clusterName = -1;

                for (int j = 0; j < numClasses; j++){

                    List<String> centroidFeatures = clusterCentroids.get(j).subList(0, numFeatures);

                    if (euclidean) {
                        double distance = MathFunction.euclideanDistance(trainingFeatures, centroidFeatures);
                        // Update which cluster point is closest, as found.
                        if (distance < minDist) {
                            minDist = distance;
                            clusterName = j; // grab classification
                        }
                    } else {
                        double distance = MathFunction.hammingDistance(trainingFeatures, centroidFeatures);
                        if (distance < minDist) {
                            minDist = distance;
                            clusterName = j;
                        }
                    }
                }
                // check to make sure clusterName is being assigned
                if (clusterName >= 0) {
                    clusters.get(clusterName).add(example);
                }
                else {
                    System.out.println("ERROR: Example not assigned to cluster");
                }

            }
            // 3) update cluster centroid locations
            int centroidNum = 0;
            for (ArrayList<ArrayList<String>> cluster: clusters
                 ) {

                ArrayList<String> updatedCentroid = new ArrayList<>();

                // if a cluster contains no values, reassign values randomly for next pass
                if (cluster.size() < 1){
                    for (int j = 0; j < numFeatures; j++){
                        updatedCentroid.add(Double.toString(Math.random()));
                    }
                    updatedCentroid.add(Integer.toString(centroidNum));
                    compareSet.add(updatedCentroid);
                    centroidNum++;
                }
                else {
                    for (int i = 0; i < numFeatures; i++) {
                        double updatedCentroidVal = 0.0;
                        for (ArrayList<String> example : cluster
                        ) {
                            updatedCentroidVal += Double.parseDouble(example.get(i));
                        }
                        updatedCentroidVal = updatedCentroidVal / cluster.size(); // mean value for that feature of examples in cluster
                        updatedCentroid.add(Double.toString(updatedCentroidVal));
                    }
                    updatedCentroid.add(Integer.toString(centroidNum));
                    compareSet.add(updatedCentroid);
                    centroidNum++;
                }
            }

            // See if clusterCentroids have changed
            if (compareSet.equals(clusterCentroids)){
                //convert arbitrary class values to real class values
                int centroidToUpdate = 0;
                for (ArrayList<ArrayList<String>> cluster: clusters
                ) {
                    Map<String, Integer> classNames = new HashMap<>();
                    for (ArrayList<String> example : cluster
                    ) {
                        String key = example.get(numFeatures);
                        // Tally up frequencies of classes in clusters (ideally, 100% one class per clusters)
                        if (classNames.containsKey(key)) {
                            int freq = classNames.get(key);
                            freq++;
                            classNames.put(key, freq);
                        } else {
                            classNames.put(key, 1);
                        }
                    }
                    int max = 0;
                    String res = ""; // class name after found
                    for (Entry<String, Integer> val : classNames.entrySet()) {
                        if (max < val.getValue()) {
                            res = val.getKey();
                            max = val.getValue();
                        }
                    }
                    clusterCentroids.get(centroidToUpdate).set(numFeatures, res);
                    centroidToUpdate++;
                }

                return KNN(clusterCentroids, testingData,1, false,true); // call to whatever other function we want.
            }
            else {
                clusterCentroids = compareSet;
                preventEndless++;
            }
        }
        System.out.println("ERROR: Something went wrong");
        return null; // should never be reached
    }
}