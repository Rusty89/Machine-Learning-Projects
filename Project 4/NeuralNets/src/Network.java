/* Network class to be used in our MLP neural networks. A Network object is comprised of Layers which is comprised
    of Nodes. Networks operate on the layers in them. They can take on any number of hidden layers and have methods
    that deal with the layers linked with in them.
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class Network
{
    private ArrayList<Layer> layers;
    public double learningRate;
    private int correctClass;
    public double regressionTarget;
    public double error;
    public ArrayList<String> guessHistory;


    public Network (int[] layerSizes, double learningRate)
    {
        this.learningRate = learningRate;
        guessHistory = new ArrayList<>();
        // create a layer for each entry in layerSizes.
        // each layer will be the size of the corresponding int in layerSizes.
        layers = new ArrayList<>();
        for (int size: layerSizes)
            layers.add(new Layer(size));

        // inform each layer of its neighbors. setPreviousLayer() also updates the previous layer's nextLayer.
        Layer previousLayer = null;
        for (Layer layer: layers)
        {
            if (previousLayer != null)
                layer.setPreviousLayer(previousLayer);
            previousLayer = layer;
        }

        // initializes the weights of nodes in each layer
        for (int a = 0; a < layers.size() - 1; a++)
            layers.get(a).initializeWeights(.005, .095);
    }

    // assumes the last entry in the ArrayList is the class and sets it as the correct answer!
    public void initializeInputLayer (ArrayList<String> values)
    {
        Layer inputLayer = layers.get(0);
        if (values.size() - 1 == inputLayer.getNodes().size())
        {
            for (int a = 0; a < values.size() - 1; a++) {
                inputLayer.getNode(a).output = Double.parseDouble(values.get(a));

                regressionTarget = Double.parseDouble(values.get(values.size() - 1));
                try {
                    correctClass = Integer.parseInt(values.get(values.size() - 1));
                } catch (NumberFormatException notAnInt) {
                    /*ignored*/
                }
            }
        }
        else {
            System.out.println("Size mismatch - input layer not initialized!");
        }
    }

    // feeds forward through the network to train
    public void feedForward()
    {
        for (int a = 1; a < layers.size(); a++)
            layers.get(a).calculateOutput();
    }

    public int getClassNumber()
    {
        return layers.get(layers.size()-1).getHighestValueNodeIndex();
    }

    // returns weather a guess was correct or not
    public boolean guessedCorrectly()
    {
        return getClassNumber() == correctClass;
    }

    // starts backpropagation for classification data sets
    public void cBackprop() {

        // Set up values in output/last layer
        Layer output = layers.get(layers.size() - 1);
        ArrayList<ArrayList<HashMap<Node, Double>>> newWeights = new ArrayList<>();

        // for each node in the output layer, check the correctness
        for (Node n: output.getNodes()) {
            n.dErr = output.getNodes().indexOf(n) == correctClass? n.output - 1: n.output;
            n.dOut = n.output * (1 - n.output);
        }

        backprop(output.getPreviousLayer(), newWeights); // perform back propagation

        // replace every connection value with the newly calculated ones
        Collections.reverse(newWeights);
        for (int a = 0; a < layers.size() - 1; a++) {
            for (int b = 0; b < layers.get(a).getNodes().size(); b++) {
                layers.get(a).getNode(b).connectionValues = newWeights.get(a).get(b);
            }
        }
    }

    // starts backpropagation for regression data sets
    public void rBackprop()
    {
        Layer output = layers.get(layers.size() - 1);
        ArrayList<ArrayList<HashMap<Node, Double>>> newWeights = new ArrayList<>();
        Node n = output.getNode(0);
        n.dErr = error =  n.output - regressionTarget;
        n.dOut = n.output * (1 - n.output);

        backprop(output.getPreviousLayer(), newWeights); // perform back propogation

        // replace every connection value with the newly calculated ones!
        Collections.reverse(newWeights);
        for (int a = 0; a < layers.size() - 1; a++) {
            for (int b = 0; b < layers.get(a).getNodes().size(); b++) {
                layers.get(a).getNode(b).connectionValues = newWeights.get(a).get(b);
            }
        }
    }

    // algorithm for performing back propagation to train our network
    private void backprop (Layer layer, ArrayList<ArrayList<HashMap<Node, Double>>> newWeights)
    {
        ArrayList<HashMap<Node, Double>> layerWeights = new ArrayList<>();
        newWeights.add(layerWeights);
        for (Node n: layer.getNodes()) {
            HashMap<Node, Double> nodeMap = new HashMap<>();
            layerWeights.add(nodeMap);

            n.dOut = n.output * (1 - n.output);
            n.dErr = 0;

            // sum up the derivative of the error
            for (Node nextNode: layer.getNextLayer().getNodes())
                n.dErr += nextNode.dErr * nextNode.dOut * n.connectionValues.get(nextNode);

            // once we have that, we can use it to calculate the new weight for each node
            for (Node nextNode: layer.getNextLayer().getNodes())
                nodeMap.put(nextNode, n.connectionValues.get(nextNode) - learningRate * (n.output * nextNode.dOut * nextNode.dErr));
        }

        if (layer.getPreviousLayer() != null) {
            backprop(layer.getPreviousLayer(), newWeights);
        }
    }

    //getter method for layers
    public ArrayList<Layer> getLayers() {
        return layers;
    }
}