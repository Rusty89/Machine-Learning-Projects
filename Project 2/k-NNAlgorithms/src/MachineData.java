import java.io.File;

public class MachineData extends Data{

    MachineData(File inputFileName) throws Exception {
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
}
