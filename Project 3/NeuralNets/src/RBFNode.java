/* Node class to be used in our RBF neural networks. Unlike the MLP algorithm, these nodes maintain a center
    that is calculated from the K-Clustering algorithms. It also maintains which layer it is apart of.
    Nodes our used to operate on data points and train our network.
 */

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
    public double momentumConstant = 0.05;

    public RBFNode (RBFLayer layer)
    {
        this.layer = layer;
    }

    // getter and setter methods for our class variables
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

    public void setBackPropChanges(int index, String weightVal) {
        backPropChanges.set(index, weightVal);
    }

    public void addBackPropChanges() {
        backPropChanges.add("0");
    }

    public void addPreviousPropChanges() {
        prevBackPropChanges.add("0");
    }

    // based on the training phase, back propogation changes are calculated and updated
    // using this function, it also uses the previous change to add momentum to the current change
    public void updateBackPropChanges(){
        for (int i = 0; i < backPropChanges.size(); i++) {
            double weight = Double.parseDouble(outputWeights.get(i));
            double previousChange = Double.parseDouble(prevBackPropChanges.get(i));
            prevBackPropChanges.set(i, backPropChanges.get(i));
            weight += Double.parseDouble(backPropChanges.get(i)) + momentumConstant * previousChange;
            String updatedWeight = weight + "";

            outputWeights.set(i, updatedWeight);
            backPropChanges.set(i, "0");
        }
    }
}