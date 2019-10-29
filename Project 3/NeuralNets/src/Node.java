import java.util.ArrayList;
import java.util.HashMap;

public class Node
{
    private Layer layer;
    private double weight;
    //private ArrayList<Double> vector;
    private HashMap<Node, Double> connectionValues;

    public Node (Layer layer, double weight)
    {
        this.layer = layer;
        this.weight = weight;
    }

    //ensures this node's weight will not be above 1 or below 0.
    public void normalize()
    {
        if (weight > 1)
            weight = 1;
        else if (weight < 0)
            weight = 0;
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
}
