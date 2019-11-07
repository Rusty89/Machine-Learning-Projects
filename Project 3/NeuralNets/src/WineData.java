/* Inherits from Data class. Defines how we pre-process the Wine data set and holds the data from it
 */

import java.io.File;

public class WineData extends Data {
    //public static int numClasses = 10;

    // constructor that reads in, normalizes, and bucketizes (for cross-validation) a data set
    WineData(File inputFileName) throws Exception {
        numClasses = 10;
        fileTo2dStringArrayList(inputFileName);
        normalizeData();
        bucketize();
    }

    @Override
    public String toString() {
        return "wine";
    }
}
