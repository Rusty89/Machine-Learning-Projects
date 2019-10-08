import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class CarData extends Data {

    CarData(File inputFileName) throws Exception{
        fileTo2dStringArrayList(inputFileName);
        preProcess();
        normalizeData();
        bucketize();
    }

    private void preProcess(){

        int numFeatures = fullSet.get(0).size();

        for (int i = 0; i < numFeatures; i++){
            Map<String, Integer> featureMap = new HashMap<>();
            int featureCount = 0;
            String featureName = "";
            for (int j = 0; j < fullSet.size(); j++){
                featureName = fullSet.get(j).get(i);
                if (featureMap.containsKey(featureName)){
                    continue;
                }
                else {
                    featureMap.put(featureName, featureCount);
                    featureCount++;
                }
            }
            for (int k = 0; k < fullSet.size(); k++){
                featureName = fullSet.get(k).get(i);
                fullSet.get(k).set(i, featureMap.get(featureName).toString());
            }
        }
    }
}
