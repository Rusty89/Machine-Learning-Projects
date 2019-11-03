/* Inherits from Data class. Defines how we pre-process the Wine data set and holds the data from it
 */

import java.io.File;

public class WineData extends Data {

    // constructor that reads in, normalizes, and bucketizes (for cross-validation) a data set
    WineData(File inputFileName) throws Exception {
        fileTo2dStringArrayList(inputFileName);
        preProcess();
        normalizeData();
        bucketize();
        findNumClassifications();
    }

    public void preProcess(){
        // normalize the classification

        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        int indexOfClass = fullSet.get(0).size()-1;
        for (int i = 0; i < fullSet.size() ; i++) {
            double classValue = Double.parseDouble(fullSet.get(i).get(indexOfClass));
            if( classValue >= max){
                max = classValue;
            }
            if(classValue <= min){
                min = classValue;
            }
        }

        for (int i = 0; i < fullSet.size(); i++) {
            double classValue = Double.parseDouble(fullSet.get(i).get(indexOfClass));
            classValue = (classValue-min)/(max-min);
            fullSet.get(i).set(indexOfClass, classValue+"");

        }
    }

    @Override
    public String toString() {
        return "wine";
    }
}
