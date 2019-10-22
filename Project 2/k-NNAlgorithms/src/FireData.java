/* Inherits from Data class. Defines how we pre-process the Fire data set and holds the data from it
 */

import java.io.File;

public class FireData extends Data {

    // constructor that reads in, pre-processes, normalizes, and bucketizes (for cross-validation) a data set
    FireData(File inputFileName) throws Exception {
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
    }
}