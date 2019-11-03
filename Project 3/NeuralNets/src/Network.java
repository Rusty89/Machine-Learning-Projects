import java.util.ArrayList;

public class Network
{
    private ArrayList<Layer> layers = new ArrayList<>();

    public Network (boolean isRBF, int[] layerSizes, ArrayList<ArrayList<String>> trainingSet,
                    ArrayList<ArrayList<String>> condensedSet)
    {


        // if the network is Radial Basis
        if (isRBF) {
            final int defaultLayerWeights = 1;
            final int defaultLayerRange = 0;
            final double minWeight = 0.05;
            final double minRange = 0.095;

            // create exactly 1 hidden layer for RBF network
            Layer inputLayer = new Layer(layerSizes[0], layerSizes[0], defaultLayerWeights, defaultLayerRange);
            Layer hiddenLayer = new Layer(layerSizes[1], layerSizes[0], minWeight, minRange);
            Layer outputLayer = new Layer(layerSizes[2], layerSizes[1], defaultLayerWeights, defaultLayerRange);

            hiddenLayer.setCenters(condensedSet); // set center values of our hidden layers

            // set previous and next layers for each layer in an RBF
            inputLayer.setNextLayer(hiddenLayer);
            hiddenLayer.setPreviousLayer(inputLayer);
            hiddenLayer.setNextLayer(outputLayer);
            outputLayer.setPreviousLayer(hiddenLayer);

            // add the three layers to the network
            layers.add(inputLayer);
            layers.add(hiddenLayer);
            layers.add(outputLayer);

        } else {

            // create a layer for each entry in layerSizes.
            // each layer will be the size of the corresponding int in layerSizes.
            layers = new ArrayList<>();
            for (int size: layerSizes)
                layers.add(new Layer(size, size, 0.05, .095));

        }



        // inform each layer of its neighbors. setPreviousLayer() also updates the previous layer's nextLayer.
        Layer previousLayer = null;
        for (Layer layer: layers)
        {
            if (previousLayer != null)
                layer.setPreviousLayer(previousLayer);
            previousLayer = layer;
        }
    }

    // classifies a training point using the RBF network
    public void classifyRBF(ArrayList<String> point) {

        Layer currentLayer = layers.get(0);

        // set the activation values for the input layer
        if (currentLayer.getPreviousLayer() == null) {

            // iterates over all nodes in the input layer
            for (int i = 0; i < currentLayer.getNodes().size(); i++) {

                // assigning an activation value to each node in the input layer
                currentLayer.getNodes().get(i).setActivationValue(Double.parseDouble(point.get(i)));
            }
        }

        // for each layer except the final output layer
        while (currentLayer.getNextLayer() != null) {

            // only for the hidden layer
            if (currentLayer.getPreviousLayer() != null && currentLayer.getNextLayer() != null) {
                currentLayer.calculateRBFActivation(); // calculate the activation values for all nodes in the hidden layer
            }

            // set variables for cleaner code below
            ArrayList<Node> nextLayerNodes = currentLayer.getNextLayer().getNodes();
            ArrayList<Node> currentLayerNodes = currentLayer.getNodes();
            int numNodesInNextLayer = nextLayerNodes.size();
            int numNodesInCurrentLayer = currentLayerNodes.size();

            // iterate over nodes in next layer to set activation values and input weights for next layer
            for (int i = 0; i < numNodesInNextLayer; i++) {
                // iterate over nodes in current layer
                for (int j = 0; j < numNodesInCurrentLayer; j++) {

                    // calculate activation values

                    // gets input weight for next layer by multiplying current nodes activation value and weight
                    double inputWeight = currentLayerNodes.get(j).getActivationValue() * currentLayerNodes.get(j).getWeight();
                    String stringifyWeight = Double.toString(inputWeight);




                    nextLayerNodes.get(i).setInputWeight(j, stringifyWeight); // set the input weight for each node
                }

            }

            currentLayer = currentLayer.getNextLayer(); // move onto the next layer
        }

        // only for the output layer
        if (currentLayer.getNextLayer() == null) {

            double maxActivationValue = 0;
            double indexOfMaxValue = 0;

            for (int i = 0; i < currentLayer.getNodes().size(); i++) {
                if (currentLayer.getNodes().get(i).getActivationValue() > maxActivationValue) {
                    maxActivationValue = currentLayer.getNodes().get(i).getActivationValue();
                    indexOfMaxValue = i;
                }
            }

            System.out.println("Activation value is " + maxActivationValue + " at index " + indexOfMaxValue);
        }
    }
}
