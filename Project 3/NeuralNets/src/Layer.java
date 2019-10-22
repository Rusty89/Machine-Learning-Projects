import java.util.ArrayList;

public class Layer
{
    private ArrayList<Node> nodes;

    public Layer (int layerSize)
    {
        nodes = new ArrayList<>();
        for (int a = 0; a < layerSize; a++)
            nodes.add(new Node(Math.random()*.095 + .05));
    }
}
