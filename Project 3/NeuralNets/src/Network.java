import java.util.ArrayList;

public class Network
{
    private ArrayList<Layer> layers;

    public Network (int[] layerSizes)
    {
        //create a layer for each entry in layerSizes.
        //each layer will be the size of the corresponding int in layerSizes.
        layers = new ArrayList<>();
        for (int size: layerSizes)
            layers.add(new Layer(size, 0.05, .095));

        //inform each layer of its neighbors. setPreviousLayer() also updates the previous layer's nextLayer.
        Layer previousLayer = null;
        for (Layer layer: layers)
        {
            if (previousLayer != null)
                layer.setPreviousLayer(previousLayer);
            previousLayer = layer;
        }

        for (int a = 0; a < layers.size() - 1; a++)
            layers.get(a).initializeWeights(.005, .095);
    }

    public void initializeInputLayer (double[] values)
    {
        Layer inputLayer = layers.get(0);
        if (values.length == inputLayer.getNodes().size())
        {
            for (int a = 0; a < values.length; a++)
                inputLayer.getNode(a).value = values[a];
        }
        else System.out.println("Size mismatch - input layer not initialized!");
    }

    public void run()
    {
        for (int a = 1; a < layers.size(); a++)
            layers.get(a).calculateOutput();
    }

    public int getClassNumber()
    {
        return layers.get(layers.size()-1).getHighestValueNodeIndex();
    }

}