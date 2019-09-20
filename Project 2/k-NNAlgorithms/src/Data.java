import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Scanner;

public class Data {
    ArrayList<ArrayList<ArrayList<Double>>> splitData= new ArrayList<ArrayList<ArrayList<Double>>>();
    ArrayList<ArrayList<Double>> fullSet = new ArrayList<ArrayList<Double>>();
    String [][] inputFileAsArray;
    String dataName;

    Data(File inputFileName, String dataName) throws Exception{
        this.dataName=dataName;
        this.inputFileAsArray=fileTo2dString(inputFileName);
        preprocessData();
        convertInputFileToArrayList();
        bucketizeData();
    }

    ///////////////////////////////////////////////////////////////
    //function for input data to be changed into a 2d string array
    private  String [][] fileTo2dString(File inputFile) throws Exception{

        //determines the dimensions of the file
        int count=0;
        Scanner sc = new Scanner(inputFile);
        while (sc.hasNextLine()){
            count++;
            sc.nextLine();
        }
        sc.close();
        sc = new Scanner(inputFile);
        String [] line= sc.nextLine().split(",");

        //once dimensions are found and 2d array data made, first line is added
        String [][] data= new String [count][line.length];
        //count is reset to increment the row while filling the array
        count=0;
        data[0]=line;
        //rest of file is filled into the 2d array
        while (sc.hasNextLine()){
            count++;
            data[count]= sc.nextLine().split(",");
        }




        //randomize the order of the input array
        //by swapping rows n times
        for (int i = 0; i < data.length ; i++) {
            //generate random number within range of data
            int randomNum= (int)Math.floor((Math.random()*data.length));

            for (int j = 0; j <data[0].length ; j++) {
                //swaps data in random row and current row
                String temp= data[i][j];
                data[i][j]=data[randomNum][j];
                data[randomNum][j]=temp;
            }
        }
        sc.close();
        return data;
    }


    private void preprocessData(){

        //switch for data input
        switch(dataName){

            case "abalone"://makes sure abalone sex data is numerical
                for (int i = 0; i <inputFileAsArray.length ; i++) {
                    switch(inputFileAsArray[i][0]){
                        case "F":
                            inputFileAsArray[i][0]="0";
                            break;
                        case "M":
                            inputFileAsArray[i][0]="1";
                            break;
                        case "I":
                            inputFileAsArray[i][0]="2";
                            break;
                    }
                }
                break;

            case "car"://changes values in car to numerical data
                for (int i = 0; i <inputFileAsArray.length; i++) {
                    for (int j = 0; j <inputFileAsArray[0].length ; j++) {
                        switch(inputFileAsArray[i][j]){
                            case "low":
                                inputFileAsArray[i][j]="0";
                                break;
                            case "med":
                                inputFileAsArray[i][j]="1";
                                break;
                            case "high":
                                inputFileAsArray[i][j]="2";
                                break;
                            case "vhigh":
                                inputFileAsArray[i][j]="3";
                                break;
                            case "small":
                                inputFileAsArray[i][j]="0";
                                break;
                            case "big":
                                inputFileAsArray[i][j]="2";
                                break;
                            case "more":
                                inputFileAsArray[i][j]="5";
                                break;
                            case "5more":
                                inputFileAsArray[i][j]="5";
                                break;
                            case "unacc":
                                inputFileAsArray[i][j]="0";
                                break;
                            case "acc":
                                inputFileAsArray[i][j]="1";
                                break;
                            case "good":
                                inputFileAsArray[i][j]="2";
                                break;
                            case "vgood":
                                inputFileAsArray[i][j]="3";
                                break;
                        }
                    }
                }
                break;
            case "forestFire":

                //i starts at one to remove first row
                for (int i = 0; i <inputFileAsArray.length; i++) {
                    for (int j = 0; j < inputFileAsArray[0].length; j++) {
                        switch (inputFileAsArray[i][j]) {
                            case "jan":
                                inputFileAsArray[i][j] = "0";
                                break;
                            case "feb":
                                inputFileAsArray[i][j] = "1";
                                break;
                            case "mar":
                                inputFileAsArray[i][j] = "2";
                                break;
                            case "apr":
                                inputFileAsArray[i][j] = "3";
                                break;
                            case "may":
                                inputFileAsArray[i][j] = "4";
                                break;
                            case "jun":
                                inputFileAsArray[i][j] = "5";
                                break;
                            case "jul":
                                inputFileAsArray[i][j] = "6";
                                break;
                            case "aug":
                                inputFileAsArray[i][j] = "7";
                                break;
                            case "sep":
                                inputFileAsArray[i][j] = "8";
                                break;
                            case "oct":
                                inputFileAsArray[i][j] = "9";
                                break;
                            case "nov":
                                inputFileAsArray[i][j] = "10";
                                break;
                            case "dec":
                                inputFileAsArray[i][j] = "11";
                                break;
                            case "sat":
                                inputFileAsArray[i][j] = "0";
                                break;
                            case "fri":
                                inputFileAsArray[i][j] = "1";
                                break;
                            case "thu":
                                inputFileAsArray[i][j] = "2";
                                break;
                            case "wed":
                                inputFileAsArray[i][j] = "3";
                                break;
                            case "tue":
                                inputFileAsArray[i][j] = "4";
                                break;
                            case "mon":
                                inputFileAsArray[i][j] = "5";
                                break;
                            case "sun":
                                inputFileAsArray[i][j] = "6";
                                break;
                        }
                    }
                }
                inputFileAsArray=inputFileAsArray;
                break;
            case "machine"://got rid of ERP value, vendor names and models
                String [][] temp2= new String[inputFileAsArray.length][inputFileAsArray[0].length-3];
                for (int i = 0; i <inputFileAsArray.length ; i++) {
                    for (int j = 2; j <inputFileAsArray[0].length -1; j++) {
                        temp2[i][j-2]=inputFileAsArray[i][j];
                    }
                }
                inputFileAsArray=temp2;
                break;
            case "segmentation":
                for (int i = 0; i <inputFileAsArray.length ; i++) {//swaps first and last column to get class on right
                    String temp=inputFileAsArray[i][0];
                    inputFileAsArray[i][0]=inputFileAsArray[i][inputFileAsArray[0].length-1];
                    inputFileAsArray[i][inputFileAsArray[0].length-1]=temp;
                    switch(inputFileAsArray[i][inputFileAsArray[0].length-1]){
                        //brickface, sky, foliage, cement, window, path, grass
                        case ("BRICKFACE"):
                            inputFileAsArray[i][inputFileAsArray[0].length-1]="0";
                            break;
                        case ("SKY"):
                            inputFileAsArray[i][inputFileAsArray[0].length-1]="1";
                            break;
                        case ("FOLIAGE"):
                            inputFileAsArray[i][inputFileAsArray[0].length-1]="2";
                            break;
                        case ("CEMENT"):
                            inputFileAsArray[i][inputFileAsArray[0].length-1]="3";
                            break;
                        case ("WINDOW"):
                            inputFileAsArray[i][inputFileAsArray[0].length-1]="4";
                            break;
                        case ("PATH"):
                            inputFileAsArray[i][inputFileAsArray[0].length-1]="5";
                            break;
                        case ("GRASS"):
                            inputFileAsArray[i][inputFileAsArray[0].length-1]="6";
                            break;

                    }
                }

                break;
            case "wine":
                for (int i = 0; i <inputFileAsArray.length ; i++) {//swaps first and last column to get class on right
                    String temp = inputFileAsArray[i][0];
                    inputFileAsArray[i][0] = inputFileAsArray[i][inputFileAsArray[0].length - 1];
                    inputFileAsArray[i][inputFileAsArray[0].length - 1] = temp;
                }
                break;
        }

    }

    private void convertInputFileToArrayList(){
        for (int i = 0; i <inputFileAsArray.length ; i++) {
            ArrayList<Double> temp= new ArrayList<Double>();
            for (int j = 0; j <inputFileAsArray[0].length; j++) {
                temp.add(Double.parseDouble(inputFileAsArray[i][j]));
            }
            fullSet.add(temp);
        }
    }

   private void bucketizeData()   {
       int start1=0;
       int start2;
       for (int i = 0; i <10 ; i++) {
           ArrayList<ArrayList<Double>> temp =new ArrayList<ArrayList<Double>>();
           for (int j = 0; j <fullSet.size()*.9 ; j++) {//generates the training sets at size 90% of data
               if(start1<fullSet.size()){
                   temp.add(fullSet.get(start1));
                   start1++;
               }else{
                   start1=0;
                   temp.add(fullSet.get(start1));
                   start1++;
               }
           }
           start2=start1;//starts the testing set at last indice of training set + 1
           splitData.add(temp);
           temp =new ArrayList<ArrayList<Double>>();
           for (int j = 0; j <fullSet.size()*.1 ; j++) {//generates the testing set of size 10% of data
               if(start2<fullSet.size()){
                   temp.add(fullSet.get(start2));
                   start2++;
               }else{
                   start2=0;
                   temp.add(fullSet.get(start2));
               }
           }
           splitData.add(temp);
       }


   }

}
