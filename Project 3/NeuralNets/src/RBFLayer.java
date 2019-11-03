import java.util.ArrayList;
import java.util.List;

public class RBFLayer
{
    private RBFLayer nextLayer;
    private RBFLayer previousLayer;
    private ArrayList<RBFNode> nodes;

    public RBFLayer (int layerSize)
    {
        nodes = new ArrayList<>();
        for (int i = 0; i < layerSize; i++) {
            RBFNode newNode = new RBFNode(this);
            nodes.add(newNode);
        }
    }

    public ArrayList<RBFNode> getNodes()
    {
        return nodes;
    }
    public RBFLayer getNextLayer()
    {
        return nextLayer;
    }
    public RBFLayer getPreviousLayer()
    {
        return previousLayer;
    }

    public void setPreviousLayer(RBFLayer layer)
    {
        previousLayer = layer;
    }

    public void setNextLayer(RBFLayer layer)
    {
        nextLayer = layer;
    }

    public void setCenters(ArrayList<ArrayList<String>> condensedSet) {
        for (int i = 0; i < nodes.size() ; i++) {
            int classificationIndexCutoff = condensedSet.get(0).size() - 1;
            List<String> centerValue = condensedSet.get(i).subList(0, classificationIndexCutoff);
            nodes.get(i).setCenter(centerValue);
        }
    }

    // use input weights and center values to calculate activation values for nodes in RBF network
    public void calculateRBFActivation() {
        // iterate over nodes in current layer
        for (int i = 0; i < getNodes().size(); i++) {
            RBFNode currentNode = getNodes().get(i);
            // run gaussian kernel activation function to calculate activation value
            double value = MathFunction.gaussianKernelActivation(currentNode.getInputWeights(), currentNode.getCenter(), 1);

            // set the activation value on the node
            getNodes().get(i).setActivationValue(value);
        }
    }

    public void calculateOutputActivation(){
        // iterate over nodes in current layer
        for (int i = 0; i < getNodes().size(); i++) {
            RBFNode currentNode = getNodes().get(i);
            // run sum the input weights
            double value = 0;
            for (int j = 0; j < currentNode.getInputWeights().size() ; j++) {
                value += Double.parseDouble(currentNode.getInputWeights().get(j));
            }
            value = value/currentNode.getInputWeights().size();
            // put output through logistic function
            value = MathFunction.logisiticActivationFunction(value);
            // set the activation value on the node
            getNodes().get(i).setActivationValue(value);
        }

    }


}
