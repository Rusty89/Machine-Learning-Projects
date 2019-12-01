import java.lang.reflect.Array;
import java.nio.channels.ClosedSelectorException;
import java.util.*;

public class DifferentialEvolution {

    ArrayList<ArrayList<String>> trainingSet;
    int [] sizes;
    Data inputData;
    final int numNetworks;
    final double crossOverRate = 0.50;
    boolean regression;

    DifferentialEvolution(Data inputData, ArrayList<ArrayList<String>> trainingSet, int sizes [], int numNetworks, boolean regression){
        this.sizes = sizes;
        this.trainingSet = trainingSet;
        this.inputData = inputData;
        this.numNetworks = numNetworks;
        this.regression = regression;
    }

    // function to train the network
    public Network evolve() throws CloneNotSupportedException {
        // creates the population of networks
        ArrayList<Network> networks = createNetworks(sizes);
        // stops after 1000 iterations of training
        int stoppingCriteria = 10;
        for (int i = 0; i < stoppingCriteria ; i++) {
            for (int j = 0; j < numNetworks; j++) {
                ArrayList<Double> mutant = createMutant(networks);
                Network trial = createTrial(mutant, networks.get(j));
                networks.set(j , selection(networks.get(j), trial));
            }
        }
        // returns the best trained network
        return findBestNetwork(networks);
    }


    // iterates over all the networks in the population and
    // returns the best network
    private Network findBestNetwork(ArrayList<Network> networks){
        Network best = networks.get(0);
        for (int i = 0; i < networks.size() ; i++) {
            if(best != networks.get(i)){
                best = selection(best, networks.get(i));
            }
        }
        // clears bests guess history so it can be used outside of training
        best.guessHistory = new ArrayList<>();
        return best;
    }


    // creates all the networks for the population
    private ArrayList<Network> createNetworks(int [] sizes){
        ArrayList<Network> networks = new ArrayList<>();
        //create an array of networks with random initializations
        for (int i = 0; i < numNetworks ; i++) {
            networks.add(new Network(sizes, 0));
        }
        return networks;
    }

    // generates a mutant vector to be used to make the trial network
    private ArrayList<Double> createMutant(ArrayList<Network> networks){
        ArrayList<Network> threeRand = drawThree(networks);
        // arraylists to hold the values of the 3 networks
        ArrayList<Double> node0Val = new ArrayList<>();
        ArrayList<Double> node1Val = new ArrayList<>();
        ArrayList<Double> node2Val = new ArrayList<>();

        // iterating over each layer of the networks
        for (int i = 0; i < threeRand.get(0).getLayers().size() ; i++) {
            ArrayList<Node> nodes0 = threeRand.get(0).getLayers().get(i).getNodes();
            ArrayList<Node> nodes1 = threeRand.get(1).getLayers().get(i).getNodes();
            ArrayList<Node> nodes2 = threeRand.get(2).getLayers().get(i).getNodes();
            // adding the weights of each connection at each node in each network
            // to their respective arraylists
            for (int j = 0; j < nodes0.size() ; j++) {
                for (Double d : nodes0.get(j).connectionValues.values()) {
                    node0Val.add(d);

                }
                for (Double d : nodes1.get(j).connectionValues.values()) {
                    node1Val.add(d);
                }
                for (Double d : nodes2.get(j).connectionValues.values()) {
                    node2Val.add(d);
                }
            }
        }

        for (int i = 0; i < node1Val.size() ; i++) {
            // x1 - x2, difference between these two vectors now stored in node1Val
            double betaConstant = .5;
            node1Val.set(i, betaConstant*((node1Val.get(i) - node2Val.get(i))));
        }

        for (int i = 0; i < node0Val.size() ; i++) {
            // add x0 and B*(x1-x2)
            node0Val.set(i, (node0Val.get(i) + node1Val.get(i)));
        }

        //return created mutant
        return node0Val;
    }


    // creates a trial network using the mutant vector
    private Network createTrial(ArrayList<Double> mutant, Network original) {
        // clones the original network to make initial trial vector
        Network trial = new Network(sizes, 0);
        for (int i = 0; i < original.getLayers().size() ; i++) {
            ArrayList<Node> trialNodes = trial.getLayers().get(i).getNodes();
            ArrayList<Node> originalNodes = original.getLayers().get(i).getNodes();
            for (int j = 0; j < trialNodes.size(); j++) {
                int finalJ = j;
                // updates the trial network with the original network values
                Collection<Double> originalNodeValues = originalNodes.get(j).connectionValues.values();
                Collection<Node> trialNodeSet = trialNodes.get(j).connectionValues.keySet();
                Iterator<Double> it = originalNodeValues.iterator();
                Iterator<Node> it2 = trialNodeSet.iterator();
                while(it.hasNext()){
                    trialNodes.get(j).connectionValues.put(it2.next(), it.next());
                }
            }
        }

        for (int i = 0; i < trial.getLayers().size() ; i++) {
            ArrayList<Node> trialNodes = trial.getLayers().get(i).getNodes();
            for (int j = 0; j < trialNodes.size(); j++) {
                int finalJ = j;
                // updates the trial network with the mutant
                trialNodes.get(j).connectionValues.forEach((key, value) -> {
                    double randVal = Math.random();
                    if(randVal < crossOverRate){
                        trialNodes.get(finalJ).connectionValues.put(key, mutant.get(0));
                    }
                    mutant.remove(0);
                });
            }
        }
        //return a trial vector
        return trial;
    }

    private Network selection(Network trial, Network original){
        original.guessHistory = new ArrayList<>();
        trial.guessHistory = new ArrayList<>();
        if(!regression){
            for (ArrayList<String> test : trainingSet) {
                original.initializeInputLayer(test);
                original.feedForward();
                original.guessHistory.add(original.getClassNumber() + "");
                trial.initializeInputLayer(test);
                trial.feedForward();
                trial.guessHistory.add(original.getClassNumber() + "");
            }
            ArrayList<String> lossOriginal = MathFunction.processConfusionMatrix(original.guessHistory, trainingSet);
            ArrayList<String> lossTrial = MathFunction.processConfusionMatrix(trial.guessHistory, trainingSet);

            if(Double.parseDouble(lossOriginal.get(2)) > Double.parseDouble(lossTrial.get(2))){
                return original;
            }else{
                return  trial;
            }
        }
        else{
            for (ArrayList<String> test : trainingSet) {
                original.initializeInputLayer(test);
                original.feedForward();
                original.guessHistory.add(original.getClassNumber() + "");
                trial.initializeInputLayer(test);
                trial.feedForward();
                trial.guessHistory.add(original.getClassNumber() + "");
            }
            String lossOriginal = MathFunction.rootMeanSquaredError(original.guessHistory, trainingSet, inputData.fullSet);
            String lossTrial = MathFunction.rootMeanSquaredError(trial.guessHistory, trainingSet, inputData.fullSet);

            if(Double.parseDouble(lossOriginal) < Double.parseDouble(lossTrial)){
                return original;
            }else{
                return  trial;
            }
        }



    }

    // draws three distinct random networks from the pool of networks and returns them
    private ArrayList<Network> drawThree(ArrayList<Network> networks){
        ArrayList<Network> selectedNetworks = new ArrayList<>();
        Random rand = new Random();
        while (selectedNetworks.size() < 3) {
            Network selected = networks.get(rand.nextInt(networks.size()));
            if(!selectedNetworks.contains(selected)){
                selectedNetworks.add(selected);
            }
        }
        return selectedNetworks;
    }

}
