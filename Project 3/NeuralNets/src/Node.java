import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Node
{
    private Layer layer;
    private double weight;
    public List<String> center;
    //private ArrayList<Double> vector;
    private HashMap<Node, Double> connectionValues;

    public Node (Layer layer, double weight)
    {
        this.layer = layer;
        this.weight = weight;
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
