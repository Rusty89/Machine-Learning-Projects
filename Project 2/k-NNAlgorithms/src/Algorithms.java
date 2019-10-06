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

        int lengthOfFeatures = trainingData.get(0).size() - 1;

        ArrayList<ArrayList<String>> KMSet = new ArrayList<>(); // Eventual final set of cluster centroids
        ArrayList<ArrayList<String>> compareSet = new ArrayList<>(); // Comparison set
        ArrayList<ArrayList<ArrayList<String>>> clusters = new ArrayList<>();

        // initialize cluster points randomly
        for (int i = 0; i < numClasses; i++) {
            ArrayList<String> point = new ArrayList<>();
            for (int j = 0; j < lengthOfFeatures; j++){
                point.add(Double.toString(Math.random())); // assign random feature values for random location
            }
            point.add(Integer.toString(i)); // arbitrary class value
            KMSet.add(point);
        }
        for (ArrayList<String> cluster: KMSet
             ) {
            //System.out.println(cluster.toString());
        }
        // Update cluster centroids until no change happens
        while (true){
            System.out.println(KMSet.get(0).size());

            // Variables
            ArrayList<ArrayList<String>> clusterValues = new ArrayList<>();
            int KMSetFeatureLastIndex = KMSet.get(0).size() - 1;

            // First we find nearest points to each cluster centroid
            for (int i = 0; i < numClasses; i++){ // for each cluster centroid
                for (ArrayList<String> example: trainingData // find all examples that are closest
                     ) {
                    List<String> trainingFeatures = example.subList(0, lengthOfFeatures - 1);
                    Double minDist = Double.MAX_VALUE;
                    Double clusterNum = 0.0;

                    // Compare each example to each cluster centroid
                    for (int j = 0; j < numClasses; j++) {
                        List<String> clusterFeatures = KMSet.get(j).subList(0, KMSetFeatureLastIndex);
                        if (euclidean) {
                            double distance = MathFunction.euclideanDistance(trainingFeatures, clusterFeatures);
                            // Update which cluster point is closest, as found.
                            if (distance < minDist) {
                                minDist = distance;
                                clusterNum = Double.parseDouble(example.get(example.size() - 1)); // grab classification
                            }
                        } else {
                            double distance = MathFunction.hammingDistance(trainingFeatures, KMSet.get(j).subList(0, KMSetFeatureLastIndex));
                            if (distance < minDist) {
                                minDist = distance;
                                clusterNum = Double.parseDouble(KMSet.get(j).get(KMSetFeatureLastIndex));
                            }
                        }
                    }
                    if (clusterNum == i){ // if the example was closest to centroid i
                        clusterValues.add(example);
                    }
                }
                clusters.add(clusterValues); // add Arraylist of examples to Arraylist of clusters
            }

            // Then we update the positions of the cluster centroids
            for (ArrayList<ArrayList<String>> clusterGroups: clusters // For each cluster
                 ) {
                ArrayList<String> newCentroid = new ArrayList<>();
                for (ArrayList<String> example: clusterGroups // for each example in a cluster group
                     ) {

                    double featureVal = 0.0;
                    for (String feature: example.subList(0, lengthOfFeatures) // for each feature in an example, excluding classIndex
                         ) {
                        featureVal += Double.parseDouble(feature);
                    }
                    double meanFeature = featureVal / (lengthOfFeatures); // mean of featureVal... i.e. k MEANS
                    String newCentroidFeature = Double.toString(meanFeature); // convert to string
                    newCentroid.add(newCentroidFeature); // build new cluster centroid values
                }
                compareSet.add(newCentroid);
            }

            // compare new cluster centroids with old to see if their was a change
            if (compareSet.equals(KMSet)){ // ideal set found, finished
                //convert arbitrary class values to real values
                int numToUpdate = 0;
                for (ArrayList<ArrayList<String>> clusterGroups: clusters
                     ) {
                    Map<String, Integer> classNames = new HashMap<>();
                    for (ArrayList<String> example: clusterGroups
                    ) {
                        String key = example.get(lengthOfFeatures);
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
                    ArrayList<String> fullCentroid = (ArrayList<String>)KMSet.get(numToUpdate).clone();
                    fullCentroid.add(res);
                    KMSet.set(numToUpdate, fullCentroid); // update KMSet to have classes of centroid
                    numToUpdate++;
                }
                return KNN(KMSet, testingData, 1, regression, euclidean);
            }
            else {
                KMSet = compareSet; // update KMSet
                compareSet.clear(); // clear compare set for future iteration
            }
        }

        /*
        // variables to hold sizes of things
        int lengthOfTrainingSet = trainingData.size();
        int lengthOfTestingSet = testingData.size();
        int lengthOfFeatures = trainingData.get(0).size() - 1;
        int classIndex = trainingData.get(0).size() - 1;

        ArrayList<ArrayList<String>> KMSet = new ArrayList<>(); // final set of cluster points (to be returned)
        ArrayList<ArrayList<String>> updatedKMSet = new ArrayList<>(); // used to see if KMSet is changing
        ArrayList<ArrayList<ArrayList<String>>> clusters = new ArrayList<>(); // Stores examples in nearest current clusters

        // initialize cluster points randomly
        for (int i = 0; i < numClasses; i++) {
            ArrayList<String> point = new ArrayList<>();
            for (int j = 0; j < lengthOfFeatures; j++){
                point.add(Double.toString(Math.random())); // assign random feature values for random location
            }
            point.add(Integer.toString(i)); // arbitrary class value
            KMSet.add(point);
            clusters.add(new ArrayList<>());
            System.out.println("Created point...");
        }

        // Until cluster points have no change
        while (true) {

            // Go through examples in training set, and see which cluster point they are closest to.
            for (ArrayList<String> example : trainingData
            ) {
                List<String> trainingFeatures = example.subList(0, lengthOfFeatures - 1);
                int KMSetFeatureLastIndex = KMSet.get(0).size() - 2;

                String clusterNum = "0";
                Double minDist = Double.MAX_VALUE;

                // Compare each example to each cluster point
                for (int i = 0; i < numClasses; i++) {
                    if (euclidean) {
                        double distance = MathFunction.euclideanDistance(trainingFeatures, KMSet.get(i).subList(0, KMSetFeatureLastIndex));
                        // Update which cluster point is closest, as found.
                        if (distance < minDist) {
                            minDist = distance;
                            clusterNum = KMSet.get(i).get(KMSetFeatureLastIndex + 1); // grab classification
                        }
                    } else {
                        double distance = MathFunction.hammingDistance(trainingFeatures, KMSet.get(i).subList(0, KMSetFeatureLastIndex));
                        if (distance < minDist) {
                            minDist = distance;
                            clusterNum = KMSet.get(i).get(KMSetFeatureLastIndex + 1);
                        }
                    }
                }
                clusters.get(Integer.parseInt(clusterNum)).add(example);
            }
            // Recalculate new clusters

            for (int i = 0; i < numClasses; i++) { // each cluster point
                ArrayList<String> updatedCluster = new ArrayList<>();
                for (int j = 0; j < clusters.get(i).get(0).size() - 1; j++){ // for each feature
                    double val = 0.0;
                    for (ArrayList<String> example: clusters.get(i)
                         ) { // for each example
                        val += Double.parseDouble(example.get(j)); // tally the feature value
                    }
                    String updatedVal = Double.toString(val / (clusters.get(i).size() - 1)); // take the mean
                    updatedCluster.add(updatedVal);
                }
                updatedCluster.add(Integer.toString(i));
                updatedKMSet.add(updatedCluster);
            }
            // Check if there's been any change to the clusters
            if (updatedKMSet.equals(KMSet)) {
                //convert arbitrary class values to real values
                for (int i = 0; i < numClasses; i++){
                    Map<String, Integer> classNames = new HashMap<>();
                    for (ArrayList<String> example: clusters.get(i)
                         ) {
                        String key = example.get(classIndex);
                        // Tally up frequencies of classes in clusters (ideally, 100% one class per clusters)
                        if (classNames.containsKey(key)){
                            int freq = classNames.get(key);
                            freq++;
                            classNames.put(key, freq);
                        }
                        else{
                            classNames.put(key, 1);
                        }
                    }
                    int max = 0;
                    String res = "";

                    for(Entry<String, Integer> val : classNames.entrySet())
                    {
                        if (max < val.getValue())
                        {
                            res = val.getKey();
                            max = val.getValue();
                        }
                    }
                    KMSet.get(i).set(classIndex, res); // set class name to actual string
                }
                return KNN(KMSet, testingData, 1, regression, euclidean);
            } else {
                KMSet = updatedKMSet;
                updatedKMSet.clear();
            }
        } */
    }
}