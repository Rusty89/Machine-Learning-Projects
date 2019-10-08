import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Algorithms {

    // base K-Nearest Neighbor algorithm
    public static ArrayList<String> KNN(ArrayList<ArrayList<String>>trainingData, ArrayList<ArrayList<String>>testingData, int k, boolean regression, boolean euclidean){

        // variables to hold sizes of things
        int lengthOfTrainingSet = trainingData.size();
        int lengthOfTestingSet = testingData.size();
        int lengthOfFeatures = trainingData.get(0).size() - 1;
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
    
    public static ArrayList<String> PAM (ArrayList<ArrayList<String>> trainingData, ArrayList<ArrayList<String>> testingData,
                                         int k, boolean regression, int totalClasses)
	{
        ArrayList<ArrayList<String>> medoids = new ArrayList<>();
        ArrayList<Cluster> clusters = new ArrayList<>();

        //pick one random point per class to center each cluster around
        int counter = 0;
        while (counter < totalClasses)
        {
            int index = (int)(Math.random()*totalClasses);
            if (!medoids.contains(trainingData.get(index)))    //if we haven't already added the point
            {
                medoids.add(trainingData.get(index));
                clusters.add(new Cluster(trainingData.get(index)));
                counter++;
            }
        }

        //loop until there is no change in medoids
        while (true)
        {
            //add each point in the training data to one of the clusters
            for (ArrayList<String> point: trainingData)
            {
                double shortestDistance = Double.MAX_VALUE;
                Cluster closestCluster = null;

                //find which cluster the point should belong to
                for (Cluster cluster: clusters)
                {
                    //euclidean distance between point in training data and medoid of the cluster
                    double distance = MathFunction.euclideanDistance(point.subList(0, point.size()-2),
                                      cluster.getMedoid().subList(0, cluster.getMedoid().size()-2));
                    if (distance < shortestDistance)
                    {
                        shortestDistance = distance;
                        closestCluster = cluster;
                    }
                }
                //after going through every cluster, add the point to the one with the nearest medoid
                closestCluster.points.add(point);
            }

            //all points are now assigned to a cluster
            //calculate the center of each cluster
            for (Cluster cluster: clusters)
            {
                ArrayList<String> average = cluster.getAverage();
                double shortestDistance = Double.MAX_VALUE;
                ArrayList<String> closestPoint = null;

                //find the point closest to the average and set it as the new medoid
                for (ArrayList<String> point: cluster.points)
                {
                    double distance = MathFunction.euclideanDistance(point.subList(0, point.size()-1), average);
                    if (distance < shortestDistance)
                    {
                        shortestDistance = distance;
                        closestPoint = point;
                    }
                }
                cluster.setMedoid(closestPoint);
            }
            //break out of the loop if all clusters have the same medoid as they did last time
            boolean anyMoved = false;
            for (Cluster cluster: clusters)
            {
                cluster.points.clear();     //clear the list of assigned points for the next iteration
                if (cluster.medoidMoved())
                    anyMoved = true;
            }
            if (!anyMoved)  //if none moved
                break;
        }

        //make a new ArrayList of just the medoid from each cluster
        medoids.clear();
        for (Cluster cluster: clusters)
            medoids.add(cluster.getMedoid());

        //send it off to KNN as the only training data and return the result
        return KNN(medoids, testingData, k, regression, true);
	}
}