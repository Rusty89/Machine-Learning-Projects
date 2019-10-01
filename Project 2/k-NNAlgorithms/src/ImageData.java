import java.io.File;

public class ImageData extends Data {

    ImageData(File inputFileName) throws Exception{
        fileTo2dStringArrayList(inputFileName);
        preProcess();
        normalizeData();
        bucketize();
    }

    private void preProcess(){
        for (int i = 0; i <fullSet.size() ; i++) {
            //remove the first first column of data in the set
            //as it is not useful information
            fullSet.get(i).remove(0);
        }
    }
}
