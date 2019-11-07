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
    }

    @Override
    public String toString() {
        return "machine";
    }
}
