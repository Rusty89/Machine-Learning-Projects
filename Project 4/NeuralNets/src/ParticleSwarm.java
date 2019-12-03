import org.w3c.dom.xpath.XPathResult;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;

public class ParticleSwarm {
    ArrayList<ArrayList<String>> trainingSet;
    int [] sizes;
    Data inputData;
    final int numNetworks;
    // constant for weight on personal Best
    double c1 = 2;
    // constant for weight on group best
    double c2 = 2;
    // velocity limits
    double velocityClampUpper = 100;
    double velocityClampLower = -100;
    // inertia
    double inertia = .5;
    boolean regression;


    double groupBestScore;
    ArrayList<Double> personalBestScores = new ArrayList<>();
    ArrayList<ArrayList<Double>> personalBests = new ArrayList<>();
    ArrayList<ArrayList<Double>> currentState = new ArrayList<>();
    ArrayList<ArrayList<Double>> velocity = new ArrayList<>();
    ArrayList<Double> groupBest = new ArrayList<>();
    ArrayList<Network> networks = new ArrayList<>();
    Network bestNet;



    ParticleSwarm(Data inputData, ArrayList<ArrayList<String>> trainingSet, int sizes [], int numNetworks, boolean regression){
        this.sizes = sizes;
        this.trainingSet = trainingSet;
        this.inputData = inputData;
        this.numNetworks = numNetworks;
        this.regression = regression;
        swarm();
    }

    private void swarm(){
        //update position of all networks
        createNetworks(sizes);
        int stoppingPoint = 1000;
        for (int i = 0; i < stoppingPoint; i++) {
            updateVelocities();
            updateNetworks();
            calculateFitness();
        }
        // sets bestNet to the best network
        bestNet = new Network(sizes, 0);
        // puts connection values back into hashmap of network
        // by iterating over the keys and states in order
        Iterator<Double> it1 = groupBest.iterator();
        for (int j = 0; j < bestNet.getLayers().size() ; j++){
            ArrayList<Node> nodes = bestNet.getLayers().get(j).getNodes();
            for (int k = 0; k < nodes.size() ; k++) {
                // updates the network with the new connection values
                Collection<Node> nodeSet = nodes.get(k).connectionValues.keySet();
                Iterator<Node> it2 = nodeSet.iterator();
                while(it2.hasNext()){
                    Node key = it2.next();
                    Double valToBeStored = it1.next();
                    nodes.get(k).connectionValues.put(key, valToBeStored);
                }
            }
        }

    }

    // creates all the networks for the population
    private void createNetworks(int [] sizes){
        //create an array of networks with random initializations
        for (int i = 0; i < numNetworks ; i++) {
            networks.add(new Network(sizes, 0));
            personalBests.add(new ArrayList<>());
            currentState.add(new ArrayList<>());
            velocity.add(new ArrayList<>());
            if(!regression){
                personalBestScores.add(0.0);
            }else{
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
                        personalBests.get(i).add(val);
                        currentState.get(i).add(val);
                        // initializes all velocities to 0;
                        velocity.get(i).add(10.0);
                        if(regression){
                            personalBestScores.set(i, Double.MAX_VALUE);
                            groupBestScore = Double.MAX_VALUE;
                        }else{
                            personalBestScores.set(i, 0.0);
                            groupBestScore = 0.0;
                        }
                    }
                }
            }
        }

        // copies the first personal best into the group best
        for (int i = 0; i < personalBests.get(0).size() ; i++) {
            groupBest.add(personalBests.get(0).get(i));
        }
    }


    private void updateVelocities(){
        for (int i = 0; i < numNetworks ; i++) {
            int sizeOfChromosome = groupBest.size();
            double r1 = Math.random();
            double r2 = Math.random();
            for (int j = 0; j < sizeOfChromosome ; j++) {
                double pBestVal = personalBests.get(i).get(j);
                double gBestVal = groupBest.get(j);
                double currentVal = currentState.get(i).get(j);
                // first term in particle swarm equation
                // for velocity, updating previous velocity times
                // inertial
                double term1 = velocity.get(i).get(j) * inertia;
                // second and third terms using the personal and group bests of a network
                // and some constants to determine change in
                // velocity

                double term2 = c1*r1*(pBestVal-currentVal);

                double term3 = c2*r2*(gBestVal - currentVal);

                // if velocity exceeds bound, clamp it to specific values
                double newVelocity = term1 + term2 + term3;
                if(newVelocity > velocityClampUpper){
                    newVelocity = velocityClampUpper;
                }else if(newVelocity < velocityClampLower){
                    newVelocity = velocityClampLower;
                }
                //update velocity with new values for this network
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
            Iterator<Double> it1 = currentState.get(i).iterator();
            for (int j = 0; j < networks.get(i).getLayers().size() ; j++){
                ArrayList<Node> nodes = currNetwork.getLayers().get(j).getNodes();
                for (int k = 0; k < nodes.size() ; k++) {
                    // updates the network with the new connection values
                    Collection<Node> nodeSet = nodes.get(k).connectionValues.keySet();
                    Iterator<Node> it2 = nodeSet.iterator();
                    while(it2.hasNext()){
                        Node key = it2.next();
                        double valToBeStored = it1.next();
                        nodes.get(k).connectionValues.put(key, valToBeStored);
                    }
                }
            }


        }

    }

    private void calculateFitness(){
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
                // check accuracy
                double score = Double.parseDouble(loss.get(2));
                // check error
                if(score > personalBestScores.get(i)){
                    // update personal best with new state
                    for (int j = 0; j < personalBests.get(i).size(); j++) {
                        personalBests.get(i).set(j, currentState.get(i).get(j));
                    }

                    personalBestScores.set(i, score);
                    if(score > groupBestScore){
                        for (int j = 0; j < groupBest.size(); j++) {
                            groupBest.set(j, currentState.get(i).get(j));
                        }

                        groupBestScore = score;
                    }
                }
                currNetwork.guessHistory.clear();
            }
            else{
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
                    if(score < groupBestScore){
                        for (int j = 0; j < groupBest.size(); j++) {
                            groupBest.set(j, currentState.get(i).get(j));
                        }

                        groupBestScore = score;
                    }
                }
                currNetwork.guessHistory.clear();
            }
        }
    }


}
