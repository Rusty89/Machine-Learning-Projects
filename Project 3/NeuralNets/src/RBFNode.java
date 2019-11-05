import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class RBFNode
{
    private RBFLayer layer;
    private List<String> center = new ArrayList<>();
    private List<String> inputWeights = new ArrayList<>();
    private List<String> outputWeights = new ArrayList<>();
    private List<String> backPropChanges = new ArrayList<>();
    private List<String> prevBackPropChanges = new ArrayList<>();
    private double activationValue;
    public double sigma = 1;
    public double momentumConstant = .05;

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
        outputWeights.add("0");
    }

    public List<String> getBackPropChanges() {
        return backPropChanges;
    }

    public void setBackPropChanges(int index, String weightVal) { backPropChanges.set(index, weightVal); }

    public void addBackPropChanges() {
        backPropChanges.add("0");
    }

    public List<String> getPreviousPropChanges() {
        return prevBackPropChanges;
    }

    public void setPreviousPropChanges(int index, String weightVal) { prevBackPropChanges.set(index, weightVal); }

    public void addPreviousPropChanges() {
        prevBackPropChanges.add("0");
    }

    public void updateBackPropChanges(){
        for (int i = 0; i <backPropChanges.size() ; i++) {
            double weight = Double.parseDouble(outputWeights.get(i));
            double previousChange = Double.parseDouble(prevBackPropChanges.get(i));
            prevBackPropChanges.set(i, backPropChanges.get(i));
            weight += Double.parseDouble(backPropChanges.get(i))+momentumConstant*previousChange;
            String updatedWeight = weight +"";

            outputWeights.set(i, updatedWeight);
            backPropChanges.set(i,"0");
        }

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
