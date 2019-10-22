import java.util.ArrayList;

public class Network
{
    private ArrayList<Layer> layers;

    public Network (int[] layerSizes)
    {
        layers = new ArrayList<>();
        for (int size: layerSizes)
            layers.add(new Layer(size));
    }
}
