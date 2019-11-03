/* A class that holds our main method for running our algorithms on the dataset. It uses the methods of the Data class
    to perform the algorithms on all data sets after reading in the data from files in the directory.
 */

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class Driver {

    public static void main(String args[])throws Exception {

        // read in our categorical sets
        Data car = new CarData(new File("../../DataSets/car.data"));
        Data abalone = new AbaloneData(new File("../../DataSets/abalone.data"));
        Data segmentation = new ImageData(new File("../../DataSets/segmentation.data"));

        // read in our regression sets, use regression and euclidean parameters for all these sets
        Data forestFire = new FireData(new File("../../DataSets/forestfires.data"));
        Data machine = new MachineData(new File("../../DataSets/machine.data"));
        Data redWine = new WineData(new File("../../DataSets/winequality-red.csv"));
        Data whiteWine = new WineData(new File("../../DataSets/winequality-white.csv"));

        // condense all data using Condensed KNN, K-Means, and K-PAM before passing to networks
        System.out.println("\nCondense all of our data: ");
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> carCondensedTrainingSets = condenseData(car, false, false);
        //ArrayList<ArrayList<ArrayList<ArrayList<String>>>> abaloneCondensedTrainingSets = condenseData(abalone, false, true);
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> segmentationCondensedTrainingSets = condenseData(segmentation, false, true);
        //ArrayList<ArrayList<ArrayList<ArrayList<String>>>> forestFireCondensedTrainingSets = condenseData(forestFire, true, true);
        //ArrayList<ArrayList<ArrayList<ArrayList<String>>>> machineCondensedTrainingSets = condenseData(machine, true, true);
        //ArrayList<ArrayList<ArrayList<ArrayList<String>>>> redWineCondensedTrainingSets = condenseData(redWine, true, true);
        //ArrayList<ArrayList<ArrayList<ArrayList<String>>>> whiteWineCondensedTrainingSets = condenseData(whiteWine, true, true);

        makeNetworks(segmentation, segmentation.dataSets.trainingSets, segmentationCondensedTrainingSets);
    }

    public static void makeNetworks(Data dataset, ArrayList<ArrayList<ArrayList<String>>> trainingSets, ArrayList<ArrayList<ArrayList<ArrayList<String>>>> condensedSets) {

        // making a Feed Forward Network



        // making the Radial Basis Networks

        ArrayList<RBFNetwork> RBNetworks = new ArrayList<>();
        final int numFeatures = trainingSets.get(0).get(0).size() - 1; // num features will not change with different training sets
        final int possibleOutcomes = dataset.numClassifications; // number of possible classifications in a data set

        // iterate over the three condensed sets: C-KNN, Kmeans, Kpam
        for (int i = 0; i < condensedSets.size(); i++) {
            // iterate over the ten training sets for each dataset
            for (int j = 0; j < condensedSets.get(i).size(); j++) {
                int condensedSetSize = condensedSets.get(i).get(j).size(); // get size of this particular condensed set
                int[] layerSizes = {numFeatures, condensedSetSize, possibleOutcomes}; // so we know how many nodes go in different layers
                RBFNetwork n = new RBFNetwork(layerSizes, condensedSets.get(i).get(j)); // build a network
                RBNetworks.add(n); // add to Radial Basis networks array for later use
                n.classifyRBF(trainingSets.get(0).get(4));
            }
        }
    }

    // method to condense to get the 3 condensed datasets for a given full dataset
    public static ArrayList<ArrayList<ArrayList<ArrayList<String>>>> condenseData(Data dataset, boolean regression, boolean euclidean) throws IOException {

        System.out.println("\nBegin "+ dataset.toString() + " tests:");
        System.out.println("--------------------\n");
        dataset.condenseSets(regression, euclidean, dataset.toString());

        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> condensedSets = new ArrayList<ArrayList<ArrayList<ArrayList<String>>>>();

        // run the condensing algorithms
        ArrayList<ArrayList<ArrayList<String>>> condensedSet = dataset.getCondensedSet();
        ArrayList<ArrayList<ArrayList<String>>> kMeansSet = dataset.getKMeansSet();
        ArrayList<ArrayList<ArrayList<String>>> kPamSet = dataset.getKPAMSet();

        condensedSets.add(condensedSet);
        condensedSets.add(kMeansSet);
        condensedSets.add(kPamSet);

        return condensedSets;
    }

    public static void printDataset(ArrayList<ArrayList<ArrayList<ArrayList<String>>>> dataset) {
        for(int i = 0; i < dataset.size(); i++) {
            System.out.println("\nCondensed set: " + i);
            System.out.println(dataset.get(i));
        }
        System.out.println();
    }
}