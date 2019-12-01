import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/*
To work with this class in the driver, we would create a new GeneticAlgorithm object, and then run
Genetic_Algorithm_Variable_Name.runAlgorithm() X amount of times in the driver. 
 */

public class GeneticAlgorithm {
    // Variables
    public ArrayList<ArrayList<String>> trainSet;
    public int[] hLayers;
    public int numClasses;
    public boolean regression;
    public ArrayList<Network> generation;

    // Tunable Parameters for the class
    public double crossoverChance = 0.20;
    public double mutationChance = 0.01;
    public int numNetworks = 10;

    // Constructor
    public GeneticAlgorithm(ArrayList<ArrayList<String>> trainSet, int[] hLayers, int numClasses, boolean regression){
        this.trainSet = trainSet;
        this.hLayers = hLayers;
        this.numClasses = numClasses;
        this.regression = regression;
        this.generation = originGen();
    }

    // Will execute each step in the GA and update the generation
    public void runAlgorithm(){
        ArrayList<Network> nextGen = new ArrayList<>();
        int i = 0;
        while (i < numNetworks/2) {
            // Selection
            ArrayList<Network> matingPair = selection();
            // Crossover
            matingPair = crossover(matingPair);
            // Mutation
            matingPair = mutation(matingPair);
            // Add offspring to next generation
            nextGen.addAll(matingPair);
            i++;
        }
        // Update generation
        this.generation = nextGen;
    }

    // Initialize the first generation with random weights
    public ArrayList<Network> originGen(){
        ArrayList<Network> gen = new ArrayList<>(numNetworks);
        for (int i = 0; i < numNetworks; i++){
            Network tempNetwork = new Network(hLayers);
            gen.add(tempNetwork);
        }
        return gen;
    }

    // Selection (tournament)
    public ArrayList<Network> selection(){
        // Save our mating pair
        ArrayList<Network> matingPair = new ArrayList<>(2);
        // Run tournament twice to produce mating pair
        for (int i = 0; i < 2; i++) {
            // draw two networks randomly (without replacement)
            Random rand = new Random();
            int randInt1 = rand.nextInt(numNetworks);
            int randInt2 = rand.nextInt(numNetworks);
            // no mating with yourself -- looking at you, Rial
            while (randInt1 == randInt2){
                randInt2 = rand.nextInt(numNetworks);
            }
            // Our two networks
            Network n1 = generation.get(randInt1);
            Network n2 = generation.get(randInt2);
            // Pick the best of the two
            // If Regression data...
            if (regression){
                // take the one with the lowest error
                if (rfitnessTest(n1) < rfitnessTest(n2)) {
                    matingPair.add(n1);
                } else {
                    matingPair.add(n2);
                }
            }
            // If Categorical data...
            else {
                // take the one with the highest accuracy
                if (cfitnessTest(n1) > cfitnessTest(n2)) {
                    matingPair.add(n1);
                } else {
                    matingPair.add(n2);
                }
            }
        }
        return matingPair;
    }

    // Crossover (Binomial Uniform)
    public ArrayList<Network> crossover(ArrayList<Network> matingPair){
        // Save offspring
        ArrayList<Network> offspring;
        int layerSize = matingPair.get(0).layers.size() - 1;

        Random rand = new Random();
        // For each layer, node in layer, connection (gene) in node...
        for (int i = 0; i < layerSize; i++) {
            for (int j = 0; j < matingPair.get(0).layers.get(i).getNodes().size(); j++) {
                HashMap<Node, Double> weights0 = matingPair.get(0).layers.get(i).getNode(j).connectionValues;
                HashMap<Node, Double> weights1 = matingPair.get(1).layers.get(i).getNode(j).connectionValues;
                for (int k = 0; k < weights0.size(); k++) {
                    // if crossover is triggered...
                    if (rand.nextDouble() < crossoverChance) {
                        Node n1 = matingPair.get(0).layers.get(i + 1).getNode(k);
                        Node n2 = matingPair.get(1).layers.get(i + 1).getNode(k);
                        // Swap values
                        // First of pair
                        weights0.put(n1, weights1.get(n2));
                        // Second of pair
                        weights1.put(n2, weights0.get(n1));
                    }
                }
            }
        }
        // return the offspring
        offspring = matingPair;
        return offspring;
    }

    // Mutation
    public ArrayList<Network> mutation(ArrayList<Network> matingPair){
        // Save the mutated offspring
        ArrayList<Network> mutOffspring;
        int layerSize = matingPair.get(0).layers.size() - 1;

        Random rand = new Random();
        // 1st in mating pair
        // Each each layer, node in layer, connection (gene) in node...
        for (int i = 0; i < layerSize; i++) {
            for (int j = 0; j < matingPair.get(0).layers.get(i).getNodes().size(); j++) {
                HashMap<Node, Double> weights = matingPair.get(0).layers.get(i).getNode(j).connectionValues;
                for (int k = 0; k < weights.size(); k++) {
                    // If mutation is triggered...
                    if (rand.nextDouble() < mutationChance) {
                        Node n1 = matingPair.get(0).layers.get(i + 1).getNode(k);
                        // Coin flip on whether mutation adds or subtracts from amount
                        if (rand.nextDouble() > 0.50){
                            weights.put(n1, weights.get(n1) + 0.1);
                        }
                        else {
                            weights.put(n1, weights.get(n1) - 0.1);
                        }
                    }
                }
            }
        }
        // Repeat process for the 2nd of the mating pair
        // Each each layer, node in layer, connection (gene) in node...
        for (int i = 0; i < layerSize; i++) {
            for (int j = 0; j < matingPair.get(1).layers.get(i).getNodes().size(); j++) {
                HashMap<Node, Double> weights = matingPair.get(1).layers.get(i).getNode(j).connectionValues;
                for (int k = 0; k < weights.size(); k++) {
                    // If mutation is triggered...
                    if (rand.nextDouble() < mutationChance) {
                        Node n2 = matingPair.get(1).layers.get(i + 1).getNode(k);
                        // Coin flip on whether mutation adds or subtracts from amount
                        if (rand.nextDouble() > 0.50){
                            weights.put(n2, weights.get(n2) + 0.1);
                        }
                        else {
                            weights.put(n2, weights.get(n2) - 0.1);
                        }
                    }
                }
            }
        }
        // Return mutated offspring
        mutOffspring = matingPair;
        return mutOffspring;
    }

    // Fitness Test(s)
    // Regression Test
    public double rfitnessTest(Network n){
        // Calculate and return absolute error
        double error = 0;
        for (ArrayList<String> example : trainSet
        ) {
            n.initializeInputLayer(example);
            n.feedForward();
            error += Math.abs(n.error);
        }
        error /= trainSet.size();
        return error;
    }

    // Categorical Test
    public double cfitnessTest(Network n){
        // Calculate and return accuracy
        double accuracy;
        for (ArrayList<String> example : trainSet
        ) {
            n.initializeInputLayer(example);
            n.feedForward();
            n.guessHistory.add(n.getClassNumber() + "");
        }
        ArrayList<String> loss = MathFunction.processConfusionMatrix(n.guessHistory, trainSet);
        accuracy = Double.parseDouble(loss.get(2));
        return accuracy;
    }
}
