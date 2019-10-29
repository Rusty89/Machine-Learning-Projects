/* Inherits from Data class. Defines how we pre-process the Wine data set and holds the data from it
 */

import java.io.File;

public class WineData extends Data {

    // constructor that reads in, normalizes, and bucketizes (for cross-validation) a data set
    WineData(File inputFileName) throws Exception {
        fileTo2dStringArrayList(inputFileName);
        normalizeData();
        bucketize();
    }

    @Override
    public String toString() {
        return "wine";
    }
}
