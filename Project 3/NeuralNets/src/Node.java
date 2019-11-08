/* Node class to be used in our MLP neural networks. A Node has weight, activation, input, output, and error values
    associated with it. It also maintains which layer it is apart of. Nodes our used to operate on data points and
    train our network.
 */

import java.util.ArrayList;
import java.util.HashMap;

public class Node
{
    private Layer layer;
    public double input;
    public double output;
    public double dErr; // errDer = error derivative (1 of 2 needed), outDer = output derivative (2 of 2 needed)
    public double dOut;
    public HashMap<Node, Double> connectionValues;

    public Node (Layer layer)
    {
        this.layer = layer;
        connectionValues = new HashMap<>();
    }

    // returns an ArrayList of the nodes in the next layer, which should be connected to this one.
    private ArrayList<Node> nextNodes()
    {
        try {
            return layer.getNextLayer().getNodes();
        }
        catch (NullPointerException noLayer) {
            return null;
        }
    }
    // returns an ArrayList of the nodes in the previous layer (for back propagation)
    private ArrayList<Node> previousNodes()
    {
        try {
            return layer.getPreviousLayer().getNodes();
        }
        catch (NullPointerException noLayer) {
            return null;
        }
    }

    // initialize the weight of the nodes
    public void initializeWeights (double minWeight, double weightRange)
    {
        for (Node node: nextNodes())
            connectionValues.put(node, Math.random()*weightRange + minWeight);
    }

    // calculate the output value coming from the node
    public void calculateOutput()
    {
        double sum = 0;
        for (Node previousNode: previousNodes())
            sum += previousNode.output * previousNode.connectionValues.get(this);
        output = MathFunction.logisticActivationFunction(sum);
        input = sum;
    }
}
