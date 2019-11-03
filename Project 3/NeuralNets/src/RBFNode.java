import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class RBFNode
{
    private RBFLayer layer;
    private List<String> center = new ArrayList<>();
    private List<String> inputWeights = new ArrayList<>();
    private List<String> outputWeights = new ArrayList<>();
    private double activationValue;
    private HashMap<Node, Double> connectionValues;

    public RBFNode (RBFLayer layer)
    {
        this.layer = layer;

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

    public List<String> getInputWeights() {
        return inputWeights;
    }

    public void setInputWeight(int index, String weightVal) {
        inputWeights.set(index, weightVal);
    }

    public void addInputWeight() {
        inputWeights.add("");
    }

    public List<String> getOutputWeights() {
        return outputWeights;
    }

    public void setOutputWeights(int index, String weightVal) {
        outputWeights.set(index, weightVal);
    }

    public void addOutputWeight() {
        outputWeights.add("");
    }



    //returns an ArrayList of the nodes in the next layer, which should be connected to this one.
    private ArrayList<RBFNode> nextNodes()
    {
        try {
            return layer.getNextLayer().getNodes();
        }
        catch (NullPointerException noLayer) {
            return null;
        }
    }
    //returns an ArrayList of the nodes in the previous layer (for back propagation)
    private ArrayList<RBFNode> previousNodes()
    {
        try {
            return layer.getPreviousLayer().getNodes();
        }
        catch (NullPointerException noLayer) {
            return null;
        }
    }


}
