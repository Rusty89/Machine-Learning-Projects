import java.io.File;

public class WineData extends Data {

    WineData(File inputFileName) throws Exception {
        fileTo2dStringArrayList(inputFileName);
        normalizeData();
        bucketize();
    }

}
