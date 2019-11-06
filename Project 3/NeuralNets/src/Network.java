import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class Network
{
    private ArrayList<Layer> layers;
    private double learningRate = .05;
    private int correctClass;

    public Network (int[] layerSizes)
    {
        //create a layer for each entry in layerSizes.
        //each layer will be the size of the corresponding int in layerSizes.
        layers = new ArrayList<>();
        for (int size: layerSizes)
            layers.add(new Layer(size, 0.05, .095));

        //inform each layer of its neighbors. setPreviousLayer() also updates the previous layer's nextLayer.
        Layer previousLayer = null;
        for (Layer layer: layers)
        {
            if (previousLayer != null)
                layer.setPreviousLayer(previousLayer);
            previousLayer = layer;
        }

        for (int a = 0; a < layers.size() - 1; a++)
            layers.get(a).initializeWeights(.005, .095);
    }

    public void initializeInputLayer (double[] values)
    {
        Layer inputLayer = layers.get(0);
        if (values.length == inputLayer.getNodes().size())
        {
            for (int a = 0; a < values.length; a++)
                inputLayer.getNode(a).output = values[a];
        }
        else System.out.println("Size mismatch - input layer not initialized!");
    }
    //assumes the last entry in the ArrayList is the class and sets it as the correct answer!
    public void initializeInputLayer (ArrayList<String> values)
    {
        Layer inputLayer = layers.get(0);
        if (values.size() - 1 == inputLayer.getNodes().size())
        {
            for (int a = 0; a < values.size() - 1; a++)
                inputLayer.getNode(a).output = Double.parseDouble(values.get(a));
            correctClass = Integer.parseInt(values.get(values.size() - 1));
        }
        else System.out.println("Size mismatch - input layer not initialized!");
    }

    public void feedForward()
    {
        for (int a = 1; a < layers.size(); a++)
            layers.get(a).calculateOutput();
    }

    public int getClassNumber()
    {
        return layers.get(layers.size()-1).getHighestValueNodeIndex();
    }
    public boolean guessedCorrectly()
    {
        return getClassNumber() == correctClass;
    }

    public void backprop(){
        // Set up values in output/last layer
        Layer output = layers.get(layers.size()- 1);
        ArrayList<ArrayList<HashMap<Node, Double>>> newWeights = new ArrayList<>();
        for (Node n: output.getNodes()) {
            n.dErr = guessedCorrectly() ? n.output - 1 : n.output;
            n.dOut = n.output * (1 - n.output);
        }
        backprop(output.getPreviousLayer(), newWeights);

        //replace every connection value with the newly calculated ones!
        Collections.reverse(newWeights);
        for (int a = 0; a < layers.size() - 1; a++)
            for (int b = 0; b < layers.get(a).getNodes().size(); b++)
                layers.get(a).getNode(b).connectionValues = newWeights.get(a).get(b);
    }

    private void backprop (Layer layer, ArrayList<ArrayList<HashMap<Node, Double>>> newWeights)
    {
        ArrayList<HashMap<Node, Double>> layerWeights = new ArrayList<>();
        newWeights.add(layerWeights);
        for (Node n: layer.getNodes()) {
            n.dOut = n.output * (1 - n.output);
            n.dErr = 0;

            HashMap<Node, Double> nodeMap = new HashMap<>();
            layerWeights.add(nodeMap);
            //sum up the derivative of the error
            for (Node nextNode: layer.getNextLayer().getNodes())
                n.dErr += nextNode.dErr * nextNode.dOut * n.connectionValues.get(nextNode);
            //once we have that, we can use it to calculate the new weight for each node!
            for (Node nextNode: layer.getNextLayer().getNodes())
                nodeMap.put(nextNode, n.connectionValues.get(nextNode) - learningRate * n.output * nextNode.dOut * nextNode.dErr);
        }
        if (layer.getPreviousLayer() != null)
            backprop(layer.getPreviousLayer(), newWeights);
    }
}