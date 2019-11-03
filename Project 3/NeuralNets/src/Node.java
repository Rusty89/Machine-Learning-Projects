import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Node
{
    private Layer layer;
    private double weight;
    private List<String> center = new ArrayList<>();
    private List<String> inputWeights = new ArrayList<>();
    private double activationValue;
    private HashMap<Node, Double> connectionValues;

    public Node (Layer layer, double weight)
    {
        this.layer = layer;
        this.weight = weight;
    }

    public List<String> getCenter() {
        return center;
    }

    public void setCenter(List<String> centerValue) {
        center = centerValue;
    }

    public double getActivationValue() {
        return activationValue;
    }

    public void setActivationValue(double value) {
        activationValue = value;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weightVal) {
        weight = weightVal;
    }

    public List<String> getInputWeights() {
        return inputWeights;
    }

    public void setInputWeight(int index, String weightVal) {
        inputWeights.set(index, weightVal);
    }

    public void addInputWeight() {
        inputWeights.add("");
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
