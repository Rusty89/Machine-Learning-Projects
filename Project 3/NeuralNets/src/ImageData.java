/* Inherits from Data class. Defines how we pre-process the Image data set and holds the data from it
 */

import java.io.File;

public class ImageData extends Data {

    // constructor that reads in, pre-processes, normalizes, and bucketizes (for cross-validation) a data set
    ImageData(File inputFileName) throws Exception{
        numClasses = 7;
        fileTo2dStringArrayList(inputFileName);
        preProcess();
        normalizeData();
        bucketize();
        findNumClassifications();
    }

    private void preProcess(){

            // converts classification strings to integers so they can be operated on more easily
            for (int i = 0; i < fullSet.size(); i++) {
                int indexOfLast = fullSet.get(i).size() - 1;

                // swap first and last column so that classifications are on the right
                String temp=fullSet.get(i).get(0);
                fullSet.get(i).set(0, fullSet.get(i).get(indexOfLast));
                fullSet.get(i).set(indexOfLast, temp);

                String classification = fullSet.get(i).get(fullSet.get(0).size() - 1);
                int classIndex = fullSet.get(0).size() - 1;

                if(classification.equals("BRICKFACE")){
                    fullSet.get(i).set(classIndex, "0");
                }
                else if(classification.equals("SKY")){
                    fullSet.get(i).set(classIndex, "1");
                }
                else if(classification.equals("FOLIAGE")){
                    fullSet.get(i).set(classIndex, "2");
                }
                else if(classification.equals("CEMENT")){
                    fullSet.get(i).set(classIndex, "3");
                }
                else if(classification.equals("WINDOW")){
                    fullSet.get(i).set(classIndex, "4");
                }
                else if(classification.equals("PATH")){
                    fullSet.get(i).set(classIndex, "5");
                }
                else{
                    fullSet.get(i).set(classIndex, "6");
                }

            }
        }

    // used to get the name of the dataset quickly for printing output
    @Override
    public String toString() {
        return "image";
    }
}
