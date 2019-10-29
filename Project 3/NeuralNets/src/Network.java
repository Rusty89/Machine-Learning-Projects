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
            layers.add(new Layer(size));

        //inform each layer of its neighbors. setPreviousLayer() also updates the previous layer's nextLayer.
        Layer previousLayer = null;
        for (Layer layer: layers)
        {
            if (previousLayer != null)
                layer.setPreviousLayer(previousLayer);
            previousLayer = layer;
        }
    }
}
