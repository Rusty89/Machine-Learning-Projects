import java.util.ArrayList;

public class Network
{
    private ArrayList<Layer> layers = new ArrayList<>();

    public Network (boolean isRBF, int[] layerSizes, ArrayList<ArrayList<String>> condensedSet)
    {
        // if the network is Radial Basis
        if (isRBF) {
            // create exactly 1 hidden layer for RBF network
            Layer inputLayer = new Layer(layerSizes[0]);
            Layer hiddenLayer = new Layer(layerSizes[1]);
            Layer outputLayer = new Layer(layerSizes[2]);

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
            // initialize weights as appropriate for RBF
            initializeWeightsRBF();

        } else {

            // create a layer for each entry in layerSizes.
            // each layer will be the size of the corresponding int in layerSizes.
            layers = new ArrayList<>();
            for (int size: layerSizes)
                layers.add(new Layer(size));

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

    public void initializeWeightsRBF(){

        final double minWeight = 0.05;
        final double minRange = 0.095;
        Layer currentLayer = layers.get(0);
        while (currentLayer.getNextLayer()!=null){
            // initialize input layer output weights
            if(currentLayer.getPreviousLayer()==null){
                for (int i = 0; i < currentLayer.getNodes().size() ; i++) {
                    for (int j = 0; j < currentLayer.getNextLayer().getNodes().size() ; j++) {
                        currentLayer.getNodes().get(i).addOutputWeight();
                        currentLayer.getNodes().get(i).setOutputWeights(j, "1");
                    }
                }
            }
            // initialize hidden layer output weights
            else{
                for (int i = 0; i < currentLayer.getNodes().size() ; i++) {
                    for (int j = 0; j < currentLayer.getNextLayer().getNodes().size() ; j++) {
                        double randWeight = Math.random()*minRange + minWeight;
                        String valOfRandWeight = randWeight+"";
                        currentLayer.getNodes().get(i).addOutputWeight();
                        currentLayer.getNodes().get(i).setOutputWeights(j, valOfRandWeight);
                    }
                    for (int j = 0; j < currentLayer.getPreviousLayer().getNodes().size() ; j++) {
                        currentLayer.getNodes().get(i).addInputWeight();
                    }
                }
            }
            currentLayer = currentLayer.getNextLayer();
        }
        for (int i = 0; i < currentLayer.getNodes().size() ; i++) {

            for (int j = 0; j < currentLayer.getPreviousLayer().getNodes().size() ; j++) {
                currentLayer.getNodes().get(i).addInputWeight();
            }

        }

    }

    // classifies a training point using the RBF network
    public double classifyRBF(ArrayList<String> point) {

        Layer currentLayer = layers.get(0);
        while(currentLayer.getNextLayer() != null){

            // set the activation values for the input layer
            if (currentLayer.getPreviousLayer() == null) {

                // iterates over all nodes in the input layer
                for (int i = 0; i < currentLayer.getNodes().size(); i++) {

                    // assigning an activation value to each node in the input layer
                    currentLayer.getNodes().get(i).setActivationValue(Double.parseDouble(point.get(i)));
                }
            }

            // for each layer except the final output layer
            if (currentLayer.getNextLayer() != null && currentLayer.getPreviousLayer() != null) {

                // only for the hidden layer
                if (currentLayer.getPreviousLayer() != null && currentLayer.getNextLayer() != null) {
                    currentLayer.calculateRBFActivation(); // calculate the activation values for all nodes in the hidden layer
                }

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
                    double weightToNextLayer = Double.parseDouble(currentLayerNodes.get(j).getOutputWeights().get(i));
                    double inputWeight = currentLayerNodes.get(j).getActivationValue() * weightToNextLayer;
                    String stringifyWeight = Double.toString(inputWeight);

                    nextLayerNodes.get(i).setInputWeight(j, stringifyWeight); // set the input weight for each node
                }

            }
            currentLayer = currentLayer.getNextLayer(); // move onto the next layer

        }

        // only for the output layer
        currentLayer.calculateOutputActivation();

        double maxActivationValue = 0;
        double indexOfMaxValue = 0;

        for (int i = 0; i < currentLayer.getNodes().size(); i++) {

            if (currentLayer.getNodes().get(i).getActivationValue() > maxActivationValue) {
                maxActivationValue = currentLayer.getNodes().get(i).getActivationValue();
                indexOfMaxValue = i;
            }
        }

        System.out.println("Activation value is " + maxActivationValue + " at index " + indexOfMaxValue);
        System.out.println("Actual class was "+ point.get(point.size()-1));
        return indexOfMaxValue;
    }


    // very much in progress, do not use
    public void trainRBFNetwork(ArrayList<ArrayList<String>> trainingData){
        int maxIterations = 10000;
        while(maxIterations>0){
            maxIterations--;
            for (int i = 0; i < trainingData.size() ; i++) {
                double classification = classifyRBF(trainingData.get(i));


            }
        }
    }
}
