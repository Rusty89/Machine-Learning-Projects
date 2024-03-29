/* A class that holds our Particle Swarm Optimization Algorithm. This is an evolutionary approach method to train our
    feedforward neural networks.
 */

import java.util.*;

public class ParticleSwarm implements Comparator<Node> {
    ArrayList<ArrayList<String>> trainingSet;
    int [] sizes;
    Data inputData;
    int numNetworks;
    int failureLimit;

    // constant for weight on personal Best
    double c1 = .5;

    // constant for weight on group best
    double c2 = 3.5;

    // velocity limits
    double velocityClampUpper = .05;
    double velocityClampLower = -0.05;

    double inertia = .9;
    boolean regression;

    int indexOfBest = 0;
    double groupBestScore;
    double prevGroupBestScore = 0;
    ArrayList<Double> personalBestScores = new ArrayList<>();
    ArrayList<ArrayList<Double>> personalBests = new ArrayList<>();
    ArrayList<ArrayList<Double>> currentState = new ArrayList<>();
    ArrayList<ArrayList<Double>> velocity = new ArrayList<>();

    // best chromosome from any generation
    ArrayList<Double> groupBest = new ArrayList<>();
    ArrayList<Network> networks = new ArrayList<>();
    Network bestNet;

    // best chromosome for current generation
    ArrayList<Double> globalBest = new ArrayList<>();


    ParticleSwarm(Data inputData) {
        this.inputData = inputData;
    }

    // training function to run through many iterations of swarming
    private void swarm(){
        //update position of all networks
        createNetworks(sizes);

        // hard stopping point to ensure termination
        int stoppingPoint = 1000;
        int noImprovementCounter = 0;
        calculateFitness();
        for (int i = 0; i < stoppingPoint; i++) {
            prevGroupBestScore = groupBestScore;
            // calculate velocities
            updateVelocities();

            // update networks with new values using the equation
            // currentState = inertia * velocity + c1*r1(pBest - currentState) + c2*r2(gBest - currentState)
            updateNetworks();

            // check fitness of new states
            calculateFitness();

            // if progress stagnates end swarming
            if(Math.abs(prevGroupBestScore - groupBestScore) < 0.001) {
                noImprovementCounter++;
            } else {
                noImprovementCounter = 0;
            }
            if(noImprovementCounter > failureLimit){
                i = stoppingPoint;
            }

        }

        // sets bestNet to the best network from all generations
        bestNet = networks.get(indexOfBest);

        // puts connection values back into hashmap of network
        // by iterating over the keys and states in order

        updateConnections(groupBest, bestNet);
    }

    // creates all the networks for the population
    private void createNetworks(int [] sizes) {

        //create an array of networks with random initializations
        for (int i = 0; i < numNetworks ; i++) {
            networks.add(new Network(sizes, 0));
            personalBests.add(new ArrayList<>());
            currentState.add(new ArrayList<>());
            velocity.add(new ArrayList<>());

            if(!regression){
                personalBestScores.add(0.0);
            } else {
                personalBestScores.add(Double.MAX_VALUE);
            }

            for (int j = 0; j < networks.get(i).getLayers().size() ; j++) {
                ArrayList<Node> nodes = networks.get(i).getLayers().get(j).getNodes();
                for (int k = 0; k < nodes.size(); k++) {

                    // updates the trial network with the original network values
                    Collection<Double> nodeValues = nodes.get(k).connectionValues.values();
                    Iterator<Double> it = nodeValues.iterator();

                    // moves all the personal bests (the initial state in this case)
                    // into an arraylist of doubles for easy use later
                    while(it.hasNext()){
                        double val = it.next();

                        // initializes positions of personal bests to
                        // random chromosomes to give a starting directions
                        double valOfRand = 0;

                        if(Math.random() > 0.5){
                            valOfRand = -Math.random();
                        }else{
                            valOfRand = +Math.random();
                        }

                        personalBests.get(i).add(valOfRand);
                        currentState.get(i).add(val);

                        // initializes all velocities
                        velocity.get(i).add(10.0);

                        if(regression) {
                            personalBestScores.set(i, Double.MAX_VALUE);
                            groupBestScore = Double.MAX_VALUE;
                        } else {
                            personalBestScores.set(i, 0.0);
                            groupBestScore = 0.0;
                        }
                    }
                }
            }
        }

        // copies the first personal best into the group best
        for (int i = 0; i < personalBests.get(0).size(); i++) {
            groupBest.add(personalBests.get(0).get(i));
        }
    }

    private void updateVelocities(){
        for (int i = 0; i < numNetworks ; i++) {
            int sizeOfChromosome = groupBest.size();

            // random values parts of terms 2 and 3
            double r1 = Math.random();
            double r2 = Math.random();

            for (int j = 0; j < sizeOfChromosome ; j++) {
                double pBestVal = personalBests.get(i).get(j);
                double gBestVal = globalBest.get(j);
                double currentVal = currentState.get(i).get(j);

                // first term in particle swarm equation
                // for velocity, updating previous velocity times inertial
                double term1 = velocity.get(i).get(j) * inertia;

                // second and third terms using the personal and group bests of a network
                // and some constants to determine change in velocity
                double term2 = c1*r1*(pBestVal-currentVal);
                double term3 = c2*r2*(gBestVal - currentVal);

                // if velocity exceeds bound, clamp it to specific values
                double newVelocity = term1 + term2 + term3;

                if(newVelocity > velocityClampUpper){
                    newVelocity = velocityClampUpper;
                }else if(newVelocity < velocityClampLower){
                    newVelocity = velocityClampLower;
                }

                // update velocity with new values for this network
                velocity.get(i).set(j, newVelocity);
            }
        }
    }

    private void updateNetworks(){
        // updates the positions of the networks based on calculated
        // velocities and the current state of the network
        for (int i = 0; i < numNetworks ; i++) {
            int sizeOfChromosome = groupBest.size();
            Network currNetwork = networks.get(i);
            for (int j = 0; j < sizeOfChromosome; j++) {
                double currentVal = currentState.get(i).get(j);
                double velocityVal = velocity.get(i).get(j);
                double newState = currentVal + velocityVal;
                currentState.get(i).set(j, newState);
            }

            // puts connection values back into hashmap of network
            // by iterating over the keys and states in order
            updateConnections(currentState.get(i), currNetwork);

        }
    }

    // calculates fitness of all networks
    private void calculateFitness(){
        double bestAtCurrentState;
        if(!regression){
            bestAtCurrentState = 0.0;

        }else{
            bestAtCurrentState = Double.MAX_VALUE;
        }
        for (int i = 0; i < numNetworks ; i++) {

            Network currNetwork = networks.get(i);
            currNetwork.guessHistory.clear();

            if(!regression){
                for (ArrayList<String> test : trainingSet) {
                    currNetwork.initializeInputLayer(test);
                    currNetwork.feedForward();
                    currNetwork.guessHistory.add(currNetwork.getClassNumber() + "");

                }
                ArrayList<String> loss = MathFunction.processConfusionMatrix(currNetwork.guessHistory, trainingSet);

                double score = Double.parseDouble(loss.get(2)); // check accuracy

                if(score > personalBestScores.get(i)){
                    // update personal best with new state
                    for (int j = 0; j < personalBests.get(i).size(); j++) {
                        personalBests.get(i).set(j, currentState.get(i).get(j));
                    }
                    personalBestScores.set(i, score);
                    // update if best score of all generations
                    if(score > groupBestScore){
                        for (int j = 0; j < groupBest.size(); j++) {
                            groupBest.set(j, currentState.get(i).get(j));
                        }
                        indexOfBest = i;
                        groupBestScore = score;

                    }
                }

                // updating best chromosome for current generation
                if(score > bestAtCurrentState){
                    globalBest = currentState.get(i);
                    bestAtCurrentState = score;
                }
                currNetwork.guessHistory.clear();
            }
            else {
                for (ArrayList<String> test : trainingSet) {
                    currNetwork.initializeInputLayer(test);
                    currNetwork.feedForward();
                    currNetwork.guessHistory.add(currNetwork.getLayers().get(sizes.length-1).getNodes().get(0).output + "");

                }
                String loss = MathFunction.rootMeanSquaredError(currNetwork.guessHistory, trainingSet, inputData.fullSet);
                double score = Double.parseDouble(loss);

                // check error
                if(score < personalBestScores.get(i)){

                    // update personal best with new state
                    for (int j = 0; j < personalBests.get(i).size(); j++) {
                        personalBests.get(i).set(j, currentState.get(i).get(j));
                    }
                    personalBestScores.set(i, score);

                    // update if best score of all generations
                    if(score < groupBestScore){
                        for (int j = 0; j < groupBest.size(); j++) {
                            groupBest.set(j, currentState.get(i).get(j));
                        }
                        indexOfBest = i;
                        groupBestScore = score;
                    }
                }

                // updating best chromosome for generation
                if(score < bestAtCurrentState){
                    globalBest = currentState.get(i);
                    bestAtCurrentState = score;
                }
                currNetwork.guessHistory.clear();
            }
        }
    }

    public  ArrayList<Double> runTest(int numNetworks, int [] sizes, int failureLimit, boolean regression, int setNum){
        // resets values prior to a run, making it ready to start
        this.sizes = sizes;
        this.trainingSet = inputData.dataSets.trainingSets.get(setNum);
        this.numNetworks = numNetworks;
        this.regression = regression;
        this.failureLimit = failureLimit;
        ArrayList<Double> results = new ArrayList<>();
        swarm(); // call to perform actual swarming

        // runs the test with the best net
        bestNet.guessHistory.clear();
        if(regression){
            for (ArrayList<String> test : inputData.dataSets.testSets.get(setNum)) {
                bestNet.initializeInputLayer(test);
                bestNet.feedForward();
                bestNet.guessHistory.add(bestNet.getLayers().get(sizes.length - 1).getNodes().get(0).output + "");
            }
            String loss = MathFunction.rootMeanSquaredError(bestNet.guessHistory, inputData.dataSets.testSets.get(setNum), inputData.fullSet);
            String loss2 = MathFunction.meanAbsoluteError(bestNet.guessHistory, inputData.dataSets.testSets.get(setNum), inputData.fullSet);
            results.add(Double.parseDouble(loss));
            results.add(Double.parseDouble(loss2));
        } else {
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

        // resets after done
        reset();
        return results;

    }

    // function to clear all values back to constructor state
    private void reset(){
        inertia = 1;
        indexOfBest = 0;
        prevGroupBestScore = 0;
        personalBestScores = new ArrayList<>();
        personalBests = new ArrayList<>();
        currentState = new ArrayList<>();
        velocity = new ArrayList<>();

        // best chromosome from any generation
        groupBest = new ArrayList<>();
        networks = new ArrayList<>();
        bestNet = null;
    }

    // regroups nodes between layers after iterating through the hashmap
    private void updateConnections(ArrayList<Double> connectionList, Network currentNetwork){

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

    @Override
    public int compare(Node a, Node b) {
        return a.id - b.id;
    }
}
