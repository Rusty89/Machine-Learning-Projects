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
        for (int i = 0; i < fullSet.size(); i++) {
            double classification = Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size() - 1));
            int classIndex = fullSet.get(0).size() - 1;
            // classification suggested by data
            if(classification <= 1){
                fullSet.get(i).set(classIndex, "0");
            }
            else if(classification <= 10){
                fullSet.get(i).set(classIndex, "1");
            }
            else if(classification <= 100){
                fullSet.get(i).set(classIndex, "2");
            }
            else if(classification <= 1000){
                fullSet.get(i).set(classIndex, "3");
            }
            else{
                fullSet.get(i).set(classIndex, "4");
            }
        }
    }

    @Override
    public String toString() {
        return "fire";
    }
}