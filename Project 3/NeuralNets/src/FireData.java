/* Inherits from Data class. Defines how we pre-process the Fire data set and holds the data from it
 */

import java.io.File;

public class FireData extends Data {
    //public static int numClasses = 5;

    // constructor that reads in, pre-processes, normalizes, and bucketizes (for cross-validation) a data set
    FireData(File inputFileName) throws Exception {
        numClasses = 5;
        fileTo2dStringArrayList(inputFileName);
        preProcess();
        normalizeData();
        bucketize();
    }

    private void preProcess() {
        for (int i = 0; i < fullSet.size(); i++) {
            /*  removes the data representing the months
                and days of the week as these will
                not be valid data points using euclidean distance */
            fullSet.get(i).remove(2);
            fullSet.get(i).remove(2);
        }

        // normalize the classification value
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        int indexOfClass = fullSet.get(0).size()-1;
        for (int i = 0; i < fullSet.size(); i++) {
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
        return "fire";
    }
}