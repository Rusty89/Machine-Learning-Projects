/* Layer class to be used in our MLP neural networks. A Network object is comprised of Layers. A Layer object is comprised
    of Nodes. Layers are connected via a linked list in a network. This class includes methods for operating
    on modes and getting/setting relevant values.
 */

import java.util.ArrayList;

public class Layer
{
    private Layer nextLayer;
    private Layer previousLayer;
    private ArrayList<Node> nodes;
    public int nodeID = 0;
    // layer constructor that adds nodes with given ranges to the layer
    public Layer (int layerSize)
    {
        nodes = new ArrayList<>();
        // adds nodes to a new layer when created
        for (int i = 0; i < layerSize; i++) {
            nodes.add(new Node(this));
            nodes.get(i).id = nodeID;
            nodeID++;
        }
    }

    // getter /setter methods
    public ArrayList<Node> getNodes()
    {
        return nodes;
    }
    public Node getNode (int index)
    {
        return nodes.get(index);
    }
    public Layer getNextLayer()
    {
        return nextLayer;
    }
    public Layer getPreviousLayer()
    {
        return previousLayer;
    }


    public void setPreviousLayer (Layer layer)
    {
        previousLayer = layer;
        layer.nextLayer = this;
    }

    public void calculateOutput()
    {
        for (Node node: nodes)
            node.calculateOutput();
    }
    public void initializeWeights (double minWeight, double weightRange)
    {
        for (Node node: nodes)
            node.initializeWeights(minWeight, weightRange);
    }

    // returns index of highest valued node
    public int getHighestValueNodeIndex()
    {
        double greatest = 0;
        int greatestIndex = -1;
        for (int a = 0; a < nodes.size(); a++)
        {
            if (nodes.get(a).output > greatest)
            {
                greatest = nodes.get(a).output;
                greatestIndex = a;
            }
        }
        return greatestIndex;
    }
}
