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
        while (sc.hasNextLine()){
            ArrayList<String> line= new ArrayList<>(Arrays.asList(sc.nextLine().split(",")));;
            fullSet.add(line);
        }
        //shuffle the input data into a random order
        Collections.shuffle(fullSet);

    }

    public void normalizeData(){
        //array lists to hold max and min values
        ArrayList<Double> max= new ArrayList<Double>();
        ArrayList<Double> min= new ArrayList<Double>();

        //initialized mins and maxes
        for (int i = 0; i <fullSet.get(0).size()-1 ; i++) {
            max.add(Double.MIN_VALUE);
            min.add(Double.MAX_VALUE);
        }
        //check down each column
        for (int j = 0; j <fullSet.get(0).size()-1; j++) {
            for (int i = 0; i <fullSet.size() ; i++) {
                //find max and mins in each column of data
                max.set(j , Double.max(max.get(j), Double.parseDouble(fullSet.get(i).get(j))));
                min.set(j , Double.min(min.get(j), Double.parseDouble(fullSet.get(i).get(j))));
            }
        }
        //go through the data set and normalize values between 0-1
        for (int i = 0; i <fullSet.size() ; i++) {

            for (int j = 0; j <fullSet.get(0).size()-1 ; j++) {
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
        for (int i = 0; i <10 ; i++) {
            //initializes new ArrayLists to store sets in the CVS structure
            dataSets.trainingSets.add(new ArrayList<ArrayList<String>>());
            dataSets.validationSets.add(new ArrayList<ArrayList<String>>());
            dataSets.testSets.add(new ArrayList<ArrayList<String>>());
            //generates a training set with 80% of the data
            for (int j = 0; j <fullSet.size()*.8 ; j++) {
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
            for (int j = 0; j <fullSet.size()*.2 ; j++) {

                //generates validation set with next 10% of data
                if(j<fullSet.size()*.1){
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
