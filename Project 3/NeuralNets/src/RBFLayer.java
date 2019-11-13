/* Layer class to be used in our RBF neural networks. A Network object is comprised of Layers. A Layer object is comprised
    of Nodes. Layers are connected via a linked list in a network. This class includes methods for operating
    on modes and getting/setting relevant values.
 */

import java.util.ArrayList;
import java.util.List;

public class RBFLayer
{
    private RBFLayer nextLayer;
    private RBFLayer previousLayer;
    private ArrayList<RBFNode> nodes;

    public double bias = .1;

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

    // sets the centers of each node, calculated using the clustering algorithms
    public void setCenters(ArrayList<ArrayList<String>> condensedSet) {
        for (int i = 0; i < nodes.size(); i++) {
            int classificationIndexCutoff = condensedSet.get(0).size() - 1;
            List<String> centerValue = condensedSet.get(i).subList(0, classificationIndexCutoff);
            nodes.get(i).setCenter(centerValue);
        }
    }


    // use input weights and center values to calculate activation values for nodes in RBF network
    // using gaussian kernel activation
    public void calculateRBFActivation() {
        // iterate over nodes in current layer
        for (int i = 0; i < getNodes().size() - 1; i++) {
            RBFNode currentNode = getNodes().get(i);

            // run gaussian kernel activation function to calculate activation value
            double value = MathFunction.gaussianKernelActivation(currentNode.getInputWeights(), currentNode.getCenter(), currentNode.sigma, bias);
            // set the activation value on the node
            getNodes().get(i).setActivationValue(value);
        }
        // sets activation of the bias node
        getNodes().get(getNodes().size() - 1).setActivationValue(.5);
    }

    // reads in the input weights to all the output nodes, calculates their activation
    // by summing the inputs and running then through a sigmoid activation
    public void calculateOutputActivation(){
        // iterate over nodes in current layer
        for (int i = 0; i < getNodes().size(); i++) {
            RBFNode currentNode = getNodes().get(i);
            // run sum the input weights
            double value = 0;
            for (int j = 0; j < currentNode.getInputWeights().size(); j++) {
                value += Double.parseDouble(currentNode.getInputWeights().get(j));

            }

            // put output through logistic function
            value = MathFunction.logisticActivationFunction(value);

            // set the activation value on the node
            getNodes().get(i).setActivationValue(value);
        }
    }

    // Sets the sigma values for all the hidden nodes, finds two nearest neighbors and takes
    // the average distance between them
    // We chose to do this based on the recommendation of this paper
    // https://perso.uclouvain.be/michel.verleysen/papers/nepl03nb.pdf
    public void findSigmaForAllNodes(){
        for (int i = 0; i < nodes.size() - 1; i++) {
            List<String> center = nodes.get(i).getCenter();
            double minDistance1 = Double.MAX_VALUE;
            double minDistance2 = Double.MAX_VALUE;
            int firstNeighbor = 0;
            for (int j = 0; j < nodes.size() - 1; j++) {
                if(i != j){
                    double distanceBetweenPoints = MathFunction.euclideanDistance(center, nodes.get(j).getCenter());
                    if(distanceBetweenPoints<minDistance1){
                        minDistance1=distanceBetweenPoints;
                        firstNeighbor = j;
                    }
                }
            }

            for (int j = 0; j < nodes.size() - 1; j++) {
                if(i != j){
                    double distanceBetweenPoints = MathFunction.euclideanDistance(center, nodes.get(j).getCenter());
                    if(distanceBetweenPoints<minDistance2 && firstNeighbor != j){
                        minDistance2 = distanceBetweenPoints;
                    }
                }
            }

            double sigmaValue = (minDistance1 + minDistance2);
            if(sigmaValue == 0){
                sigmaValue = 0.1;
            }
            nodes.get(i).sigma = sigmaValue;
        }
    }
}