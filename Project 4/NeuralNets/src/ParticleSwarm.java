import org.w3c.dom.xpath.XPathResult;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

public class ParticleSwarm {
    ArrayList<ArrayList<String>> trainingSet;
    int [] sizes;
    Data inputData;
    final int numNetworks;
    double c1 = 0.90;
    double c2 = 0.50;
    double velocityClampUpper = 1;
    double velocityClampLower = -1;
    double inertia = 0.50;
    boolean regression;

    ArrayList<Double> personalBestScores;
    ArrayList<ArrayList<Double>> personalBests;
    ArrayList<ArrayList<Double>> currentState;
    ArrayList<ArrayList<Double>> velocity;
    ArrayList<Double> groupBest;
    ArrayList<Network> networks;
    Network bestNet;
    double bestFitness;


    ParticleSwarm(Data inputData, ArrayList<ArrayList<String>> trainingSet, int sizes [], int numNetworks, boolean regression){
        this.sizes = sizes;
        this.trainingSet = trainingSet;
        this.inputData = inputData;
        this.numNetworks = numNetworks;
        this.regression = regression;
        if(regression){
            bestFitness = Double.MAX_VALUE;
        }else{
            bestFitness = 0;
        }

    }

    private void swarm(){
        //update position of all networks
        createNetworks(sizes);
        int stoppingPoint = 100;
        for (int i = 0; i < stoppingPoint; i++) {
            updateVelocities();
            updateNetworks();
            calculateFitness();
        }
    }

    // creates all the networks for the population
    private void createNetworks(int [] sizes){
        //create an array of networks with random initializations
        for (int i = 0; i < numNetworks ; i++) {
            networks.add(new Network(sizes, 0));
            for (int j = 0; j < networks.get(i).getLayers().size() ; i++) {
                ArrayList<Node> nodes = networks.get(i).getLayers().get(j).getNodes();
                for (int k = 0; k < nodes.size(); k++) {
                    // updates the trial network with the original network values
                    Collection<Double> nodeValues = nodes.get(j).connectionValues.values();
                    Iterator<Double> it = nodeValues.iterator();
                    // moves all the personal bests (the initial state in this case)
                    // into an arraylist of doubles for easy use later
                    while(it.hasNext()){
                        double val = it.next();
                        personalBests.get(i).add(val);
                        currentState.get(i).add(val);
                        // initializes all velocities to 0;
                        velocity.get(i).add(0.0);
                        if(regression){
                            personalBestScores.add(Double.MAX_VALUE);
                        }else{
                            personalBestScores.add(0.0);
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
            for (int j = 0; j < sizeOfChromosome; j++) {
                double currentVal = currentState.get(i).get(j);
                double velocityVal = velocity.get(i).get(j);
                double newState = currentVal + velocityVal;
                currentState.get(i).set(j, newState);
            }
        }
    }

    private void calculateFitness(){
        for (int i = 0; i < numNetworks ; i++) {

        }

    }


}
