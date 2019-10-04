import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

public class Data {

    public ArrayList<ArrayList<String>> fullSet = new ArrayList<>();
    public CVS dataSets = new CVS();


    public void fileTo2dStringArrayList(File inputFile) throws Exception{
        //read in input  file as an array list
        Scanner sc = new Scanner(inputFile);
        int maxCount=1000;//max number of lines of data, to keep test manageable
        while (sc.hasNextLine()){
            ArrayList<String> line= new ArrayList<>(Arrays.asList(sc.nextLine().split(",")));;
            fullSet.add(line);

        }
        //shuffle the input data into a random order
        Collections.shuffle(fullSet);

        //remove items to make dataset manageable
        while(fullSet.size()>maxCount){
            fullSet.remove(0);
        }

    }

    public void normalizeData(){
        //array lists to hold max and min values
        ArrayList<Double> max= new ArrayList<Double>();
        ArrayList<Double> min= new ArrayList<Double>();
        int indexOfLastTrait = fullSet.get(0).size()-1;
        int sizeOfSet = fullSet.size();
        //initialized mins and maxes
        for (int i = 0; i <indexOfLastTrait ; i++) {
            max.add(Double.MIN_VALUE);
            min.add(Double.MAX_VALUE);
        }
        //check down each column
        for (int j = 0; j <indexOfLastTrait; j++) {
            for (int i = 0; i <sizeOfSet ; i++) {
                //find max and mins in each column of data
                max.set(j , Double.max(max.get(j), Double.parseDouble(fullSet.get(i).get(j))));
                min.set(j , Double.min(min.get(j), Double.parseDouble(fullSet.get(i).get(j))));
            }
        }
        //go through the data set and normalize values between 0-1

        for (int i = 0; i <sizeOfSet; i++) {
            for (int j = 0; j <indexOfLastTrait; j++) {
                //if max and min are the same, normalize it to a 1
                if((max.get(j)-min.get(j))==0){
                    fullSet.get(i).set(j, "1.0");
                }else{
                    //normalize using equation
                    //(x-min)/(max-min)
                    fullSet.get(i).set(j , (Double.parseDouble(fullSet.get(i).get(j))-min.get(j))/(max.get(j)-min.get(j)) +"");
                }

            }
        }

    }

    public void bucketize(){

        int countTrainingSet=0;
        double eightyPercentOfData = 0.8 * fullSet.size();
        double twentyPercentOfData = 0.2 * fullSet.size();
        double tenPercentOfData = 0.1 *fullSet.size();
        for (int i = 0; i <10 ; i++) {
            //initializes new ArrayLists to store sets in the CVS structure
            dataSets.trainingSets.add(new ArrayList<ArrayList<String>>());
            dataSets.validationSets.add(new ArrayList<ArrayList<String>>());
            dataSets.testSets.add(new ArrayList<ArrayList<String>>());
            //generates a training set with 80% of the data
            for (int j = 0; j <eightyPercentOfData ; j++) {
                if (countTrainingSet < fullSet.size()) {
                    dataSets.trainingSets.get(i).add(fullSet.get(countTrainingSet));
                    countTrainingSet++;
                }
                else{
                    countTrainingSet=0;
                    dataSets.trainingSets.get(i).add(fullSet.get(countTrainingSet));
                    countTrainingSet++;
                }
            }
            int countValidationAndTest=countTrainingSet;
            for (int j = 0; j <twentyPercentOfData ; j++) {

                //generates validation set with next 10% of data
                if(j<tenPercentOfData){
                    if (countValidationAndTest < fullSet.size()) {
                        dataSets.validationSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    }
                    else{
                        countValidationAndTest=0;
                        dataSets.validationSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    }
                }
                //generates test sets with the last 10% of data
                else{
                    if (countValidationAndTest < fullSet.size()) {
                        dataSets.testSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    }
                    else{
                        countValidationAndTest=0;
                        dataSets.testSets.get(i).add(fullSet.get(countValidationAndTest));
                        countValidationAndTest++;
                    }
                }
            }
        }
    }
}
