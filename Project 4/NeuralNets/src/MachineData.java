/* Inherits from Data class. Defines how we pre-process the Machine data set and holds the data from it
 */

import java.io.File;

public class MachineData extends Data{

    // constructor that reads in, pre-processes, normalizes, and bucketizes (for cross-validation) a data set
    MachineData(File inputFileName) throws Exception {
        numClasses = 5;
        fileTo2dStringArrayList(inputFileName);
        preProcess();
        normalizeData();
        bucketize();
        findNumClassifications();
    }

    private void preProcess() {
        for (int i = 0; i < fullSet.size(); i++) {
            /*  remove the first two columns of data in the set
                makes and models are not relevant to a regression
                classification */
            fullSet.get(i).remove(0);
            fullSet.get(i).remove(0);
        }

        // normalize the class value
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        int indexOfClass = fullSet.get(0).size() - 1;
        for (int i = 0; i < fullSet.size(); i++) {
            double classValue = Double.parseDouble(fullSet.get(i).get(indexOfClass));

            if (classValue >= max) {
                max = classValue;
            }
            if (classValue <= min) {

                min = classValue;
            }
        }

        for (int i = 0; i < fullSet.size(); i++) {
            double classValue = Double.parseDouble(fullSet.get(i).get(indexOfClass));
            classValue = (classValue - min) / (max - min);
            fullSet.get(i).set(indexOfClass, classValue + "");
        }
    }

    // used to get the name of the dataset quickly for printing output
    @Override
    public String toString() {
        return "machine";
    }
}
