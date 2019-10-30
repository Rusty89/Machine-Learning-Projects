/* A class that holds the implementation of each of the five algorithms: KNN, Edited KNN, Condensed KNN, K-Means, and
    two PAM implementations. These are the actual algorithms, as opposed to the run methods in the Data class
    that define the routine with running the data through the algorithms.
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.lang.Math;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

// a class that contains an implementation for each of the KNN algorithms, a K-Means algorithm, and two PAM algorithms
public class KNNAlgorithms {

    // implementation of Condensed KNN algorithm
    public static ArrayList<ArrayList<String>> CondensedKNN(ArrayList<ArrayList<String>>trainingData, boolean euclidean) {

        // initialize empty set
        ArrayList<ArrayList<String>> condensedTrainingData=new ArrayList<>();
        int currSizeOfCondensed = - 1;
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
        return condensedTrainingData;
    }

    // implementation of the K-Means algorithm
    public static ArrayList<ArrayList<String>> Kmeans(ArrayList<ArrayList<String>>trainingData, int numClasses){

        // local variables
        ArrayList<ArrayList<String>> clusterCentroids = new ArrayList<>(); // Holds the found cluster centroids
        int numFeatures= trainingData.get(0).size() - 1; // number of features (not index)

        // 1) initialize cluster centroids randomly
        for (int i = 0; i < numClasses; i++){ // Number of cluster centroids = number of classes
            clusterCentroids.add(MathFunction.randomCentroid(numFeatures));
        }

        // Until no change of clusterCentroid from one iteration to the next
        while (true) {

            // Variables
            ArrayList<ArrayList<String>> compareSet = new ArrayList<>(); // The new cluster centroid
            ArrayList<ArrayList<ArrayList<String>>> clusters = new ArrayList<>(); // Associated clusters to centroids

            // Initialize clusters
            for (int i = 0; i < numClasses; i++){
                clusters.add(new ArrayList<>());
            }

            // 2) Compare example distances to centroids and assign to appropriate clusters
            for (ArrayList<String> example: trainingData
            ) {

                // local variables
                List<String> trainingFeatures = example.subList(0, numFeatures);
                double minDist = Double.MAX_VALUE;
                int clusterName = -1;

                for (int j = 0; j < numClasses; j++){

                    // local variables
                    List<String> centroidFeatures = clusterCentroids.get(j).subList(0, numFeatures);
                    double distance = MathFunction.euclideanDistance(trainingFeatures, centroidFeatures);

                    // Update which cluster point is closest, as found.
                    if (distance < minDist) {
                        minDist = distance;
                        clusterName = j; // grab classification
                    }
                }
                // check to make sure clusterName is being assigned
                if (clusterName >= 0) {
                    clusters.get(clusterName).add(example);
                }
                else {
                    System.out.println("ERROR: Example not assigned to cluster"); // Error msg
                }
            }

            // 3) update cluster centroid locations
            int centroidNum = 0; // keep track of which centroid we're working with

            for (ArrayList<ArrayList<String>> cluster: clusters) {
                ArrayList<String> updatedCentroid = new ArrayList<>();

                // if a cluster contains no examples, reassign location of the related centroid to a random alternative centroid location
                if (cluster.size() < 1){
                    continue;
                }
                // for clusters with examples (common)
                else {
                    for (int i = 0; i < numFeatures; i++) {

                        double updatedCentroidVal = 0.0; // for calculating averages

                        for (ArrayList<String> example : cluster
                        ) {
                            updatedCentroidVal += Double.parseDouble(example.get(i));
                        }
                        updatedCentroidVal = updatedCentroidVal / cluster.size(); // mean value for that feature of examples in cluster
                        updatedCentroid.add(Double.toString(updatedCentroidVal)); // keep class variable consistent when updated
                    }
                    updatedCentroid.add(Integer.toString(centroidNum));
                    compareSet.add(updatedCentroid);
                    centroidNum++;
                }
            }

            // remove any centroid that has no related cluster in final set
            for (int i = 0; i < numClasses; i++){
                if (clusters.get(i).size() < 1){
                    clusters.remove(i);
                    clusterCentroids.remove(i);
                    numClasses--;
                    i--; // to repeat on index
                }
            }

            // See if clusterCentroids have changed
            if (compareSet.equals(clusterCentroids)){
                // remove any centroid that has no related cluster in final set
                for (int i = 0; i < numClasses; i++){
                    if (clusters.get(i).size() < 1){
                        clusters.remove(i);
                        clusterCentroids.remove(i);
                        numClasses--;
                        i--; // to repeat on index
                    }
                }

                // convert arbitrary class values to real class values
                int centroidToUpdate = 0;

                for (ArrayList<ArrayList<String>> cluster: clusters) {

                    Map<String, Integer> classNames = new HashMap<>(); // track frequency of class names in clusters

                    for (ArrayList<String> example : cluster) {
                        String key = example.get(numFeatures);

                        // tally up frequencies of classes in clusters (ideally, 100% one class per clusters)
                        if (classNames.containsKey(key)) {
                            int freq = classNames.get(key);
                            freq++; // increase tally
                            classNames.put(key, freq);
                        } else {
                            classNames.put(key, 1);
                        }
                    }

                    int max = 0;
                    String res = ""; // class name after found

                    // find the most common class in the cluster
                    for (Entry<String, Integer> val : classNames.entrySet()) {
                        if (max < val.getValue()) {
                            res = val.getKey();
                            max = val.getValue();
                        }
                    }

                    // set the centroid that represents the most common class in the related cluster
                    clusterCentroids.get(centroidToUpdate).set(numFeatures, res); //
                    centroidToUpdate++;
                }

                return clusterCentroids;
            }
            // otherwise, keep repeating
            else {
                clusterCentroids = compareSet; // update cluster centroids
            }
        }
    }

    // custom implementation of the Partitioning Around Medoids algorithm
    // this version calculates the true average of the cluster and finds the closest data point to it
    // this runtime is much faster, but makes a small variation from the proper PAM
    public static ArrayList<ArrayList<String>> AlternativePAM(ArrayList<ArrayList<String>> trainingData, int totalClasses)
	{
        ArrayList<ArrayList<String>> medoids = new ArrayList<>();
        ArrayList<Cluster> clusters = new ArrayList<>();

        // pick one random point per class to center each cluster around
        int counter = 0;
        while (counter < totalClasses)
        {
            int index = (int)(Math.random()*totalClasses);

            // if we haven't already added the point
            if (!medoids.contains(trainingData.get(index)))
            {
                medoids.add(trainingData.get(index));
                clusters.add(new Cluster(trainingData.get(index)));
                counter++;
            }
        }

        // loop until there is no change in medoids
        while (true)
        {
            // add each point in the training data to one of the clusters
            for (ArrayList<String> point: trainingData)
            {
                double shortestDistance = Double.MAX_VALUE;
                Cluster closestCluster = null;

                // find which cluster the point should belong to
                for (Cluster cluster: clusters)
                {
                    // euclidean distance between point in training data and medoid of the cluster
                    try{
                        double distance = MathFunction.euclideanDistance(point.subList(0, point.size()-1),
                                cluster.getMedoid().subList(0, cluster.getMedoid().size()-1));

                        if (distance < shortestDistance)
                        {
                            shortestDistance = distance;
                            closestCluster = cluster;
                        }

                    } catch(Exception e) {
                        // something must have been bad about this cluster for this error to occur
                        System.out.println(cluster);
                        clusters.remove(cluster);
                    }

                }

                /* after going through every cluster, add the point to the one with the nearest medoid
                    making sure that point doesn't already exist within the cluster */
                if(!closestCluster.points.contains(point)){
                    closestCluster.points.add(point);
                }
            }

            // all points are now assigned to a cluster
            // calculate the center of each cluster
            for (Cluster cluster: clusters)
            {
                ArrayList<String> average = cluster.getAverage();
                double shortestDistance = Double.MAX_VALUE;
                ArrayList<String> closestPoint = null;

                // find the point closest to the average and set it as the new medoid
                for (ArrayList<String> point: cluster.points)
                {
                    double distance = MathFunction.euclideanDistance(point.subList(0, point.size() - 1), average);
                    if (distance < shortestDistance)
                    {
                        shortestDistance = distance;
                        closestPoint = point;
                    }
                }
                cluster.setMedoid(closestPoint);
            }

            // break out of the loop if all clusters have the same medoid as they did last time
            boolean anyMoved = false;
            for (Cluster cluster: clusters)
            {
                cluster.points.clear();     // clear the list of assigned points for the next iteration

                if (cluster.medoidMoved())
                    anyMoved = true;
            }

            if (!anyMoved)
                break;
        }

        // make a new ArrayList of just the medoid from each cluster
        medoids.clear();
        for (Cluster cluster: clusters)
            medoids.add(cluster.getMedoid());

        // return the medoids. KNN will be called using them inside the Driver.
        return medoids;
	}
}

