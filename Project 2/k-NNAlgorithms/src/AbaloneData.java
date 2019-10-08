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

        for (int i = 0; i <fullSet.size() ; i++) {
            Double classification = Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size()-1));
            //if num rings less than 10, set to class 0, young
            if(classification<10){
                fullSet.get(i).set(i,"0");
            }
            //if num rings greater than or equal to 10 but less than 20, set to class 1, intermediate
            else if(classification<20){
                fullSet.get(i).set(i,"1");
            }
            //if num rings greater than 20, set to class 2, old
            else{
                fullSet.get(i).set(i,"2");
            }
        }
    }
}
