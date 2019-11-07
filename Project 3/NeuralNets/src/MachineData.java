/* Inherits from Data class. Defines how we pre-process the Machine data set and holds the data from it
 */

import java.io.File;

public class MachineData extends Data{
    //public static int numClasses = 5;

    // constructor that reads in, pre-processes, normalizes, and bucketizes (for cross-validation) a data set
    MachineData(File inputFileName) throws Exception {
        numClasses = 5;
        fileTo2dStringArrayList(inputFileName);
        preProcess();
        normalizeData();
        bucketize();
    }

    private void preProcess() {
        for (int i = 0; i < fullSet.size(); i++) {
            /*  remove the first two columns of data in the set
                makes and models are not relevant to a regression
                classification */
            fullSet.get(i).remove(0);
            fullSet.get(i).remove(0);
        }
        for (int i = 0; i < fullSet.size(); i++) {
            double classification = Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size() - 1));
            int classIndex = fullSet.get(0).size() - 1;
            // classification suggested by data
            if(classification <= 20){
                fullSet.get(i).set(classIndex, "0");
            }
            else if(classification <= 100){
                fullSet.get(i).set(classIndex, "1");
            }
            else if(classification <= 200){
                fullSet.get(i).set(classIndex, "2");
            }
            else if(classification <= 300){
                fullSet.get(i).set(classIndex, "3");
            }
            else{
                fullSet.get(i).set(classIndex, "4");
            }
        }
    }

    @Override
    public String toString() {
        return "machine";
    }
}
