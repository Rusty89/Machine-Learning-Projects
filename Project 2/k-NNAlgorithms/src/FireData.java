import java.io.File;

public class FireData extends Data {

    FireData(File inputFileName) throws Exception {
        fileTo2dStringArrayList(inputFileName);
        preProcess();
        normalizeData();
        bucketize();
    }

    private void preProcess() {
        for (int i = 0; i < fullSet.size(); i++) {
            //removes the data representing the months
            //and days of the week as these will
            //not be valid data points using euclidean distance
            fullSet.get(i).remove(2);
            fullSet.get(i).remove(2);
        }
    }
}