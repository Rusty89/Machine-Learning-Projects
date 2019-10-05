import java.io.File;

public class AbaloneData extends Data {

    AbaloneData(File inputFileName) throws Exception{
        fileTo2dStringArrayList(inputFileName);
        preProcess();
        normalizeData();
        bucketize();
    }

    private void preProcess(){
        for (int i = 0; i <  fullSet.size(); i++) {
            /*  remove the first first column of data in the set
                Male, Female and Infant distances cannot be resolved
                in a meaningful way */
            fullSet.get(i).remove(0);
        }
    }
}
