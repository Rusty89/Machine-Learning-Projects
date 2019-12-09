/* A class that holds our Differential Evolution algorithm. This is an evolutionary approach method to train our 
    feedforward neural networks.
 */

import java.util.*;

public class DifferentialEvolution  implements Comparator<Node>{

    ArrayList<ArrayList<String>> trainingSet;
    int [] sizes;
    Data inputData;
    int numNetworks;
    final double crossOverRate = 0.9;
    final double betaConstant = 0.05;
    final int stoppingCriteria = 100;
    boolean regression;
    Network bestNet;

    DifferentialEvolution(Data inputData) {
        this.inputData = inputData;
    }
    // function to train the network
    public Network evolve() {

        // creates the population of networks
        ArrayList<Network> networks = createNetworks(sizes);

        // stops after a certain number of iterations of training
        for (int i = 0; i < stoppingCriteria ; i++) {
            for (int j = 0; j < numNetworks; j++) {
                ArrayList<Double > mutant = createMutant(networks);
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
        // arraylists to hold the values of the 3 networks to make a mutant with
        ArrayList<Double> node0Val = createChromosome(threeRand.get(0));
        ArrayList<Double> node1Val = createChromosome(threeRand.get(1));;
        ArrayList<Double> node2Val = createChromosome(threeRand.get(2));;
        
        for (int i = 0; i < node1Val.size() ; i++) {
            // B(x1 - x2), difference between these two vectors now stored in node1Val
            // multiplied by a tunable constant
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
    private Network createTrial(ArrayList<Double> mutant, Network currentNetwork) {
        ArrayList<Double> currNetworkChromosome = createChromosome(currentNetwork);
        for (int i = 0; i < mutant.size() ; i++) {
            double crossoverChance = Math.random();
            if(crossoverChance > crossOverRate){
                mutant.set(i, currNetworkChromosome.get(i));
            }
        }
        Network trial = new Network(sizes, 0);
        updateConnections(mutant, trial);

        //return a trial vector
        return trial;
    }

    // selects if the trial or original network is better
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
                trial.guessHistory.add(trial.getClassNumber() + "");
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
                original.guessHistory.add(original.getLayers().get(sizes.length-1).getNodes().get(0).output + "");
                trial.initializeInputLayer(test);
                trial.feedForward();
                trial.guessHistory.add(trial.getLayers().get(sizes.length-1).getNodes().get(0).output + "");
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

    private ArrayList<Double> createChromosome(Network n){
        ArrayList<Double> result = new ArrayList<>();

        // iterating over each layer of the networks
        for (int i = 0; i < n.getLayers().size() ; i++) {
            ArrayList<Node> nodes0 = n.getLayers().get(i).getNodes();
            nodes0.sort(this::compare);

            // adding the weights of each connection at each node in each network
            // to their respective arraylists
            for (int j = 0; j < nodes0.size() ; j++) {
                Collection<Node> node0Collection = nodes0.get(j).connectionValues.keySet();
                Iterator<Node> node0Iterator = node0Collection.iterator();
                ArrayList<Node> node0Array = new ArrayList<>();
                while(node0Iterator.hasNext()){
                    node0Array.add(node0Iterator.next());
                }

                node0Array.sort(this::compare);
                node0Iterator = node0Array.iterator();
                while(node0Iterator.hasNext()){
                    result.add(nodes0.get(j).connectionValues.get(node0Iterator.next()));
                }

            }
        }
        return result;
    }

    // regroups nodes between layers after iterating through the hashmap
    public void updateConnections(ArrayList<Double> connectionList, Network currentNetwork){
        Iterator<Double> iterator1 = connectionList.iterator();
        for (int j = 0; j < currentNetwork.getLayers().size() ; j++){
            ArrayList<Node> nodes = currentNetwork.getLayers().get(j).getNodes();
            for (int k = 0; k < nodes.size() ; k++) {
                
                // updates the network with the new connection values
                Collection<Node> nodeSet = nodes.get(k).connectionValues.keySet();
                Iterator<Node> iterator2 = nodeSet.iterator();
                ArrayList<Node> nodeSet2 = new ArrayList<>();
                
                while(iterator2.hasNext()){
                    nodeSet2.add(iterator2.next());
                }
                
                nodeSet2.sort(this::compare);
                iterator2 = nodeSet2.iterator();
                while(iterator2.hasNext()){
                    Node key = iterator2.next();
                    double valToBeStored = iterator1.next();
                    nodes.get(k).connectionValues.put(key, valToBeStored);
                }
            }
        }
    }
    
    public  ArrayList<Double> runTest(int numNetworks, int [] sizes, boolean regression, int setNum) {
        
        // resets values prior to a run, making it ready to start
        this.sizes = sizes;
        this.trainingSet = inputData.dataSets.trainingSets.get(setNum);
        this.numNetworks = numNetworks;
        this.regression = regression;

        ArrayList<Double> results = new ArrayList<>();
        bestNet = evolve();

        // runs the test with the best network
        bestNet.guessHistory.clear();
        if(regression){
            for (ArrayList<String> test : inputData.dataSets.testSets.get(setNum)) {
                bestNet.initializeInputLayer(test);
                bestNet.feedForward();
                bestNet.guessHistory.add(bestNet.getLayers().get(sizes.length-1).getNodes().get(0).output + "");
            }
            String loss = MathFunction.rootMeanSquaredError(bestNet.guessHistory, inputData.dataSets.testSets.get(setNum), inputData.fullSet);
            String loss2 = MathFunction.meanAbsoluteError(bestNet.guessHistory, inputData.dataSets.testSets.get(setNum), inputData.fullSet);
            results.add(Double.parseDouble(loss));
            results.add(Double.parseDouble(loss2));
        }else{
            for (ArrayList<String> test : inputData.dataSets.testSets.get(0)) {
                bestNet.initializeInputLayer(test);
                bestNet.feedForward();
                bestNet.guessHistory.add(bestNet.getClassNumber() + "");
            }
            ArrayList<String> loss = MathFunction.processConfusionMatrix(bestNet.guessHistory, inputData.dataSets.testSets.get(0));

            results.add(Double.parseDouble(loss.get(0)));
            results.add(Double.parseDouble(loss.get(1)));
            results.add(Double.parseDouble(loss.get(2)));
        }
        return results;
    }

    
    @Override
    public int compare(Node a, Node b) {
        return a.id - b.id;
    }
}
