import java.util.ArrayList;

public class Layer
{
    private Layer nextLayer, previousLayer;
    private ArrayList<Node> nodes;

    public Layer (int layerSize, double minWeight, double weightRange)
    {
        nodes = new ArrayList<>();
        //TODO instead of each node having a random weight, should each connection between nodes have a random weight instead?
        for (int a = 0; a < layerSize; a++)
            nodes.add(new Node(this, Math.random()*weightRange + minWeight));
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
    //also sets the previous layer's nextLayer as this.
    public void setPreviousLayer (Layer layer)
    {
        previousLayer = layer;
        layer.nextLayer = this;
    }
}
