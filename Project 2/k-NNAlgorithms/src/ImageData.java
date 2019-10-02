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
            int indiceOfLast= fullSet.get(i).size()-1;
            //swap first and last column so that
            //classifications are on the right
            String temp=fullSet.get(i).get(0);
            fullSet.get(i).set(0, fullSet.get(i).get(indiceOfLast));
            fullSet.get(i).set(indiceOfLast, temp);

        }
    }
}
