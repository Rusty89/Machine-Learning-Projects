import java.util.ArrayList;
import java.util.List;

public class Layer
{
    private Layer nextLayer;
    private Layer previousLayer;
    private ArrayList<Node> nodes;

    public Layer (int layerSize,int sizeOfInputs, double minWeight, double weightRange)
    {
        nodes = new ArrayList<>();
        //TODO instead of each node having a random weight, should each connection between nodes have a random weight instead?
        for (int a = 0; a < layerSize; a++) {
            Node newNode = new Node(this, Math.random() * weightRange + minWeight);

            for (int i = 0; i < sizeOfInputs; i++) {
                newNode.addInputWeight();
            }
            nodes.add(newNode);
        }
    }

    public ArrayList<Node> getNodes()
    {
        return nodes;
    }
    public Layer getNextLayer()
    {
        return nextLayer;
    }
    public Layer getPreviousLayer()
    {
        return previousLayer;
    }

    public void setPreviousLayer(Layer layer)
    {
        previousLayer = layer;
    }

    public void setNextLayer(Layer layer)
    {
        nextLayer = layer;
    }

    public void setCenters(ArrayList<ArrayList<String>> condensedSet) {
        for (int i = 0; i < nodes.size() - 1; i++) {
            int classificationIndexCutoff = condensedSet.get(0).size() - 1;
            List<String> centerValue = condensedSet.get(i).subList(0, classificationIndexCutoff);
            nodes.get(i).setCenter(centerValue);
        }
    }

    // use input weights and center values to calculate activation values for nodes in RBF network
    public void calculateRBFActivation() {
        // iterate over nodes in current layer
        for (int i = 0; i < getNodes().size(); i++) {
            Node currentNode = getNodes().get(i);

            // run gaussian kernel activation function to calculate activation value
            double value = MathFunction.gaussianKernelActivation(currentNode.getInputWeights(), currentNode.getCenter(), 1);

            // set the activation value on the node
            getNodes().get(i).setActivationValue(value);
        }
    }


}
