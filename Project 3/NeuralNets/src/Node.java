import java.util.ArrayList;
import java.util.HashMap;

public class Node
{
    private Layer layer;
    public double weight, input, output, dErr, dOut; // errDer = error derivative (1 of 2 needed), outDer = output derivative (2 of 2 needed)
    public HashMap<Node, Double> connectionValues;


    public Node (Layer layer, double weight)
    {
        this.layer = layer;
        this.weight = weight;
        connectionValues = new HashMap<>();
    }

    //returns an ArrayList of the nodes in the next layer, which should be connected to this one.
    private ArrayList<Node> nextNodes()
    {
        try {
            return layer.getNextLayer().getNodes();
        }
        catch (NullPointerException noLayer) {
            return null;
        }
    }
    //returns an ArrayList of the nodes in the previous layer (for back propagation)
    private ArrayList<Node> previousNodes()
    {
        try {
            return layer.getPreviousLayer().getNodes();
        }
        catch (NullPointerException noLayer) {
            return null;
        }
    }

    public void initializeWeights (double minWeight, double weightRange)
    {
        for (Node node: nextNodes())
            connectionValues.put(node, Math.random()*weightRange + minWeight);
    }
    public void calculateOutput()
    {
        double sum = 0;
        for (Node previousNode: previousNodes())
            sum += previousNode.output * previousNode.connectionValues.get(this);
        output = MathFunction.logisticActivationFunction(sum);
        input = sum;
    }
}
