/* Inherits from Data class. Defines how we pre-process the Image data set and holds the data from it
 */

import java.io.File;

public class ImageData extends Data {

    // constructor that reads in, pre-processes, normalizes, and bucketizes (for cross-validation) a data set
    ImageData(File inputFileName) throws Exception{
        fileTo2dStringArrayList(inputFileName);
        preProcess();
        normalizeData();
        bucketize();
        findNumClassifications();
    }

    private void preProcess(){
        for (int i = 0; i < fullSet.size(); i++) {
            int indexOfLast = fullSet.get(i).size() - 1;

            //swap first and last column so that classifications are on the right
            String temp=fullSet.get(i).get(0);
            fullSet.get(i).set(0, fullSet.get(i).get(indexOfLast));
            fullSet.get(i).set(indexOfLast, temp);
        }
    }

    @Override
    public String toString() {
        return "image";
    }
}
