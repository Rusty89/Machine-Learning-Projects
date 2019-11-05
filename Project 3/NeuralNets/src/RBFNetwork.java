import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class RBFNetwork
{
    private ArrayList<RBFLayer> layers = new ArrayList<>();

    public RBFNetwork (int[] layerSizes, ArrayList<ArrayList<String>> condensedSet)
    {

        // create exactly 1 hidden layer for RBF network
        RBFLayer inputLayer = new RBFLayer(layerSizes[0]);
        RBFLayer hiddenLayer = new RBFLayer(layerSizes[1]);
        RBFLayer outputLayer = new RBFLayer(layerSizes[2]);

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

        // inform each layer of its neighbors. setPreviousLayer() also updates the previous layer's nextLayer.
        RBFLayer previousLayer = null;
        for (RBFLayer layer: layers)
        {
            if (previousLayer != null)
                layer.setPreviousLayer(previousLayer);
            previousLayer = layer;
        }
    }

    public void initializeWeightsRBF(){

        final double minWeight = 0.01;
        final double minRange = .95;
        RBFLayer currentLayer = layers.get(0);
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

                        currentLayer.getNodes().get(i).addBackPropChanges();

                        currentLayer.getNodes().get(i).addPreviousPropChanges();
                    }
                    for (int j = 0; j < currentLayer.getPreviousLayer().getNodes().size() ; j++) {
                        currentLayer.getNodes().get(i).addInputWeight();
                    }
                }
            }
            currentLayer.findSigmaForAllNodes();
            currentLayer = currentLayer.getNextLayer();
        }
        for (int i = 0; i < currentLayer.getNodes().size() ; i++) {

            for (int j = 0; j < currentLayer.getPreviousLayer().getNodes().size() ; j++) {
                currentLayer.getNodes().get(i).addInputWeight();
            }

        }

    }

    // classifies a training point using the RBF network
    public double classifyRBF(ArrayList<String> point, boolean categorical) {
        if(categorical){
            RBFLayer currentLayer = layers.get(0);
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
                ArrayList<RBFNode> nextLayerNodes = currentLayer.getNextLayer().getNodes();
                ArrayList<RBFNode> currentLayerNodes = currentLayer.getNodes();
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

            return indexOfMaxValue;


        }else{
            RBFLayer currentLayer = layers.get(0);
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
                ArrayList<RBFNode> nextLayerNodes = currentLayer.getNextLayer().getNodes();
                ArrayList<RBFNode> currentLayerNodes = currentLayer.getNodes();
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

            double output = currentLayer.getNodes().get(0).getActivationValue();

            return output;

        }

    }


    // training the RBF networks for categorical and regression classifications
    public void trainRBFNetwork(ArrayList<ArrayList<String>> trainingData, ArrayList<ArrayList<String>> orignalFullSet, double learningRate, boolean categorical){
        if(categorical){

            int maxIterations = 300;
            while(maxIterations>0){
                maxIterations--;
                RBFLayer outputLayer = layers.get(2);
                RBFLayer hiddenLayer = layers.get(1);
                Collections.shuffle(trainingData);
                ArrayList<String> classifications = new ArrayList<>();
                // for categorical data

                for (int i = 0; i < trainingData.size()/4 ; i++) {
                    int indexOfClass = trainingData.get(0).size()-1;
                    classifications.add(((int)classifyRBF(trainingData.get(i), true)+""));
                    double actualClassification = Double.parseDouble(trainingData.get(i).get(indexOfClass));
                    for (int j = 0; j < outputLayer.getNodes().size(); j++) {
                        double activationAtOutput = outputLayer.getNodes().get(j).getActivationValue();

                        if(j == actualClassification){
                            for (int k = 0; k < hiddenLayer.getNodes().size() ; k++) {
                                RBFNode hiddenNode = hiddenLayer.getNodes().get(k);
                                RBFNode outputNode = outputLayer.getNodes().get(j);
                                double dk = -(1 - activationAtOutput)*activationAtOutput*(1-activationAtOutput);
                                double activationFromNodeHiddenNodeK = Double.parseDouble(outputNode.getInputWeights().get(k));
                                double backPropChange = Double.parseDouble(hiddenNode.getBackPropChanges().get(j));
                                double weightChange = backPropChange - (dk*learningRate*activationFromNodeHiddenNodeK);
                                hiddenNode.setBackPropChanges(j,weightChange+"");
                            }

                        }
                        else{
                            for (int k = 0; k < hiddenLayer.getNodes().size() ; k++) {
                                RBFNode hiddenNode = hiddenLayer.getNodes().get(k);
                                RBFNode outputNode = outputLayer.getNodes().get(j);
                                double dk = -(0 - activationAtOutput)*activationAtOutput*(1-activationAtOutput);
                                double activationFromNodeHiddenNodeK = Double.parseDouble(outputNode.getInputWeights().get(k));
                                double backPropChange = Double.parseDouble(hiddenNode.getBackPropChanges().get(j));
                                double weightChange = backPropChange - (dk*learningRate*activationFromNodeHiddenNodeK);
                                hiddenNode.setBackPropChanges(j,weightChange+"");
                            }
                        }
                    }
                }
                for (int i = 0; i < hiddenLayer.getNodes().size() ; i++) {
                    hiddenLayer.getNodes().get(i).updateBackPropChanges();
                }
                ArrayList<String> result = new ArrayList<>();
                //result = MathFunction.processConfusionMatrix(classifications,trainingData);
                //System.out.println(result);
            }

        }else{

            int maxIterations = 300;
            while(maxIterations>0){
                maxIterations--;
                RBFLayer outputLayer = layers.get(2);
                RBFLayer hiddenLayer = layers.get(1);
                Collections.shuffle(trainingData);
                ArrayList<String> classifications = new ArrayList<>();

                // for regression data

                for (int i = 0; i < trainingData.size()/4; i++) {
                    int indexOfClass = trainingData.get(0).size()-1;

                    classifications.add((int)classifyRBF(trainingData.get(i), false)+"");

                    double activationAtOutput = outputLayer.getNodes().get(0).getActivationValue();
                    double target = Double.parseDouble(trainingData.get(i).get(indexOfClass));

                    for (int k = 0; k < hiddenLayer.getNodes().size() ; k++) {
                        RBFNode hiddenNode = hiddenLayer.getNodes().get(k);
                        RBFNode outputNode = outputLayer.getNodes().get(0);
                        double dk = -(target - activationAtOutput)*activationAtOutput*(1-activationAtOutput);
                        double activationFromNodeHiddenNodeK = Double.parseDouble(outputNode.getInputWeights().get(k));
                        double backPropChange = Double.parseDouble(hiddenNode.getBackPropChanges().get(0));
                        double weightChange = backPropChange - (dk*learningRate*activationFromNodeHiddenNodeK);

                        hiddenNode.setBackPropChanges(0,weightChange+"");
                    }

                }
                for (int i = 0; i < hiddenLayer.getNodes().size() ; i++) {
                    hiddenLayer.getNodes().get(i).updateBackPropChanges();
                }

                //String result = MathFunction.rootMeanSquaredError(classifications, trainingData, orignalFullSet);
                //System.out.println(result);
            }
        }
    }
}