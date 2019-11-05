import java.util.ArrayList;

public class Layer
{
    private Layer nextLayer, previousLayer;
    private ArrayList<Node> nodes;

    public Layer (int layerSize, double minWeight, double weightRange)
    {
        nodes = new ArrayList<>();
        for (int a = 0; a < layerSize; a++)
            nodes.add(new Node(this, Math.random()*weightRange + minWeight));
    }

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
    //also sets the previous layer's nextLayer as this.
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
    public int getHighestValueNodeIndex()
    {
        double greatest = 0;
        int greatestIndex = -1;
        for (int a = 0; a < nodes.size(); a++)
        {
            if (nodes.get(a).value > greatest)
            {
                greatest = nodes.get(a).value;
                greatestIndex = a;
            }
        }
        return greatestIndex;
    }
    public boolean isHighestValueNode(int index){
        int greatestIndex = getHighestValueNodeIndex();
        if (index == greatestIndex) {
            return true;
        }
        return false;
    }
}
