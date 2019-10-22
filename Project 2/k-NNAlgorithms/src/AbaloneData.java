/* Inherits from Data class. Defines how we pre-process the Abalone data set and holds the data from it
 */

import java.io.File;

public class AbaloneData extends Data {

    // constructor that reads in, pre-processes, normalizes, and bucketizes (for cross-validation) a data set
    AbaloneData(File inputFileName) throws Exception{
        fileTo2dStringArrayList(inputFileName);
        preProcess();
        normalizeData();
        bucketize();
    }

    private void preProcess(){
        for (int i = 0; i < fullSet.size(); i++) {
            /*  remove the first first column of data in the set
                Male, Female and Infant distances cannot be resolved
                in a meaningful way */
            fullSet.get(i).remove(0);
        }

        for (int i = 0; i < fullSet.size(); i++) {
            double classification = Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size() - 1));
            double firstThird = 8;
            double secondThird = 12;
            int classIndex = fullSet.get(0).size() - 1;
            
            // if num rings less than 8, set to class 0, young
            if(classification < firstThird){
                fullSet.get(i).set(classIndex, "0");
            }
            // if num rings greater than or equal to 8 but less than 12, set to class 1, intermediate
            else if(classification < secondThird){
                fullSet.get(i).set(classIndex, "1");
            }
            // if num rings greater than 11, set to class 2, old
            else{
                fullSet.get(i).set(classIndex, "2");
            }
        }
    }
}
