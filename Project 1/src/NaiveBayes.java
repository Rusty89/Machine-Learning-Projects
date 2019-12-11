
//////////////////////////////////////////////////////////////////////
//Authors:  Rusty Clayton, Nick Hager, Rial Johnson, Jay Van Alstyne
//Date: 9/10/2019
/////////////////////////////////////////////////////////////////////


import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Scanner;
import java.util.*;
import java.util.concurrent.TimeUnit;


public class NaiveBayes {
    private static PrintWriter printer;

    public static void main(String[] args) throws Exception {
        StartNaiveBayes("Project 1/breast cancer/breast-cancer-wisconsin.data", "cancer");
        StartNaiveBayes("Project 1/voting records/house-votes-84.data", "votes");
        StartNaiveBayes("Project 1/iris/iris.data", "iris");
        StartNaiveBayes("Project 1/soybean/soybean-small.data", "beans");
        StartNaiveBayes("Project 1/glass/glass.data", "glass");
    }

    private static void StartNaiveBayes(String fileName, String dataType) throws Exception{
        double [][]preNoiseResult;
        double [][]postNoiseResult;
        FileWriter filer = new FileWriter(dataType + " results.txt");
        printer = new PrintWriter(filer);

        System.out.println("----------------------------------------------------------------------------");
        System.out.println("Begin new set for "+ dataType + ".");
        printer.println("Begin new set for "+ dataType + ".");
        System.out.println("Reading in file...");
        TimeUnit.MILLISECONDS.sleep(1000);
        String [][] rawInput=fileTo2dString(new File(fileName));

        System.out.println("Processing data...");
        TimeUnit.MILLISECONDS.sleep(1000);
        String [][] processedInput=preprocessData(rawInput, dataType);
        printer.println("\n----------------------------------------------------------------------------\n");

        System.out.println("Printing results...");
        TimeUnit.MILLISECONDS.sleep(1000);
        preNoiseResult=processResult(crossValidation(processedInput, dataType), dataType);

        System.out.println("Adding noise to this set...");
        printer.println("\nAdding noise to this set...");
        TimeUnit.MILLISECONDS.sleep(1000);
        addNoise(processedInput);
        printer.println("\n----------------------------------------------------------------------------\n");

        System.out.println("Printing shuffled results...");
        TimeUnit.MILLISECONDS.sleep(1000);
        postNoiseResult=processResult(crossValidation(processedInput, dataType), dataType);
        printer.println("\n----------------------------------------------------------------------------\n");

        System.out.println("Calculating statistical significance...");
        //compare results to determine statistical significance of results
        significance(preNoiseResult,postNoiseResult);
        System.out.println("----------------------------------------------------------------------------\n\n");

        printer.close();
        filer.close();
    }

    ///////////////////////////////////////////////////////////////////////
    ////////function for finding out statistical significance
    private static void significance(double[][]preNoise,double[][]postNoise){
        final int numberOfLossFunctions=3;
        final double tTestBoundary=2.262;
        final int indexOfAverageResult=10;
        double [] SDpreNoise=new double[numberOfLossFunctions];
        double [] SDpostNoise= new double[numberOfLossFunctions];
        double [] t = new double [numberOfLossFunctions];
        //calculate Standard deviation and t score within the data
        for (int i = 0; i <numberOfLossFunctions ; i++) {
            //prenoise data
            double sumOfDifferenceOfSquares1=0;
            //postNoise data
            double sumOfDifferenceOfSquares2=0;
            for (int j = 0; j <10 ; j++) {
                sumOfDifferenceOfSquares1+=Math.pow(preNoise[j][i]-preNoise[indexOfAverageResult][i],2);
                sumOfDifferenceOfSquares2+=Math.pow(postNoise[j][i]-postNoise[indexOfAverageResult][i],2);
            }
            ///1/9 = 1/n-1 where n is the number of samples
            SDpreNoise[i]=(1.0/9.0)*sumOfDifferenceOfSquares1;
            SDpostNoise[i]=(1.0/9.0)*sumOfDifferenceOfSquares2;

            //calculate t

            t[i]=(preNoise[indexOfAverageResult][i]-postNoise[indexOfAverageResult][i])/((SDpreNoise[i]/Math.sqrt(10))+(SDpostNoise[i]/Math.sqrt(10)));
        }

        //using a t-table found at https://www.medcalc.org/manual/t-distribution.php
        //set p<0.05 we choose the t-value 2.262 as our cutoff point using 9 degrees of freedom
        //smallest of (n1-1) and (n2-1)  (9).  If a t score is >2.262 or < -2.262 then the
        // addition of noise was statistically significant to that loss function
        //(p<0.05 chosen because it is fairly standard in statistics)
        for (int i = 0; i < numberOfLossFunctions; i++) {
            if(t[i]<=-tTestBoundary || t[i]>=tTestBoundary){
                switch(i){
                    case(0):
                        printer.println("The difference in recall pre and post noise was statistically" +
                                " significant to p value <0.05");
                        break;
                    case(1):
                        printer.println("The difference in precision pre and post noise was statistically" +
                                " significant to p value <0.05");
                        break;
                    case(2):
                        printer.println("The difference in accuracy pre and post noise was statistically" +
                                " significant to p value <0.05");
                        break;

                }


            }else{
                switch(i){
                    case(0):
                        printer.println("The difference in recall pre and post noise was NOT statistically" +
                                " significant to p value <0.05");
                        break;
                    case(1):
                        printer.println("The difference in precision pre and post noise was NOT statistically" +
                                " significant to p value <0.05");
                        break;
                    case(2):
                        printer.println("The difference in accuracy pre and post noise was NOT statistically" +
                                " significant to p value <0.05");
                        break;

                }

            }
            printer.println("T score: "+ t[i]+ ", Degrees of freedom: 9");
        }
    }

    private static void addNoise(String [][] inputData){
        int tenPercent = (int)Math.round((0.1* (double)(inputData[0].length-1)));
        if (tenPercent<1){
            tenPercent=1;
        }

        List featureList = new ArrayList(); //store which features we are scrambling
        List alreadyShuffledColumn = new ArrayList(); //ensure we don't scramble the same features multiple times.

        for (int j = 0; j < tenPercent; j++) { //for tenPercent number of times...

            int randomNum = (int)Math.floor((Math.random()* (inputData[0].length - 1))); //...randomly select which column (feature) to shuffle
            while (alreadyShuffledColumn.contains(randomNum)){
                randomNum = (int)Math.floor((Math.random()* (inputData[0].length - 1))); //re-roll if already shuffled column (features) is found
            }
            alreadyShuffledColumn.add(randomNum); // add column # (feature) to list of alreadyshuffledcolumn for future reference (no repeats).

            for (int k = 0; k < inputData.length; k++){ //go through each example in data...
                featureList.add(inputData[k][randomNum]); //...and collect the feature to be shuffled
            }

            printer.println("The shuffled feature was column: " + randomNum);
            Collections.shuffle(featureList); //shuffle the collected feature

            for (int i = 0; i < inputData.length; i++){ //place shuffled data back into array
                inputData[i][randomNum] = featureList.get(i).toString();
            }
            featureList.clear();
        }
    }

    private static double[][] processResult(double [][][] input, String dataType){
        //output array for results of all 10 tests, and for the
        //averaged results
        final int numLossFunctions=3;
        final int numTestsPlusAvg=11;
        double [][] output=new double[numTestsPlusAvg][numLossFunctions];
        double recallAvg=0;
        double accuracyAvg=0;
        double precisionAvg=0;
        double errorAvg=0;

        for (int i = 0; i <10 ; i++) {
            double truePos=0;
            double total=0;
            double totalPos=0;
            double Rmacro=0;
            double Pmacro=0;
            double Emacro=0;
            double Amacro=0;
            double numClasses=0;
            boolean found=false;

            //find the number of present classes in the current data set
            for (int j = 0; j <input[0].length ; j++) {
                for (int k = 0; k <input[0].length; k++){
                    if(input[i][j][k] !=0 && found==false){
                        numClasses++;
                        found=true;
                    }
                }
                found=false;
            }

            for (int j = 0; j <input[0].length ; j++) {
                //gathers the diagonal column of the array, the true predictions
                truePos=input[i][j][j];
                //gathers truePos cumulatively
                totalPos+=truePos;

                double falseNeg=0;
                double falsePos=0;
                double trueNeg=0;
                for (int k = 0; k <input[0].length ; k++) {
                    //total grabs all results cumulatively
                    total+=input[i][j][k];
                    //finds trueNeg
                    trueNeg+=input[i][j][j];
                    //falseNeg grabs the row data (including the truePos results)
                    falseNeg+=input[i][j][k];
                    //falsePos grabs the column data (including the truePos results)
                    falsePos+=input[i][k][j];
                }
                //eliminate the truePos results from the falseNeg and falsePos and trueNegresults
                falseNeg-=truePos;
                falsePos-=truePos;
                trueNeg-=truePos;
                if(truePos==0){
                    //do nothing to prevent 0/0 error
                }else{
                    Rmacro+=truePos/(truePos+falseNeg);
                    Pmacro+=truePos/(truePos+falsePos);
                    Amacro+=(truePos+trueNeg)/(truePos+trueNeg+falsePos+falseNeg);
                }
            }

            //finish calculating recall and precision by dividing by number of classes
            Rmacro/=numClasses;
            Pmacro/=numClasses;
            Amacro/=numClasses;

            printer.println("Recall of test "+ (i+1) + " is: " + Rmacro);
            printer.println("Precision of test "+ (i+1) + " is: " + Pmacro);
            printer.println("Accuracy of test "+ (i+1) + " is: " + Amacro);
            printer.println();

            output[i][0]=Rmacro;
            output[i][1]=Pmacro;
            output[i][2]=Amacro;

            recallAvg+=Rmacro;
            precisionAvg+=Pmacro;
            accuracyAvg+=Amacro;
        }

        printer.println("Average Recall is    : " + recallAvg/10);
        printer.println("Average Precision is : " + precisionAvg/10);
        printer.println("Average Accuracy is  : " + accuracyAvg/10);

        output[10][0]=recallAvg/10;
        output[10][1]=precisionAvg/10;
        output[10][2]=accuracyAvg/10;

        return output;
    }

    private static double [][][] crossValidation(String [][]processedInput, String dataSet){

        ArrayList<String> classification=getClassifications(processedInput);

        ///runs tenfold cross validation by incrementing what index it draws the training and test
        //sets from then runs them through the naive bayes algorithm 10 times
        double [][][]output=new double[10][classification.size()][classification.size()];
        double trainingSetSize=0.9;
        double testingSetSize=0.1;
        int currentIndex=0;
        int beginNewTrainingSet=0;

        for (int a = 0; a <10 ; a++) {

            String[][] trainingSet = new String[(int) (processedInput.length *trainingSetSize)][processedInput[0].length];

            for (int i = 0; i <(int) (processedInput.length *.9) ; i++) {
                for (int j = 0; j <processedInput[0].length ; j++) {
                    trainingSet[i][j]=processedInput[currentIndex][j];
                }
                currentIndex++;
                if(currentIndex>processedInput.length-1){
                    currentIndex=0;
                }
            }

            beginNewTrainingSet=currentIndex;
            String[][] testingSet = new String[(int)(processedInput.length * testingSetSize)][processedInput[0].length];
            for (int i = 0; i < (int) (processedInput.length *.1); i++) {
                for (int j = 0; j <processedInput[0].length ; j++) {
                    testingSet[i][j]=processedInput[currentIndex][j];
                }
                currentIndex++;
                if(currentIndex>processedInput.length-1){
                    currentIndex=0;
                }
            }

            output[a]=NaiveBayesAlgorithm(trainingSet,testingSet, classification);
            currentIndex=beginNewTrainingSet;
        }
        return output;
    }

    private static double [][] NaiveBayesAlgorithm(String [][]trainingSet, String[][]testingSet, ArrayList<String>classification){
        int [][][] values=trainingAlgorithm(trainingSet,classification);

        double fail=0;
        double success=0;
        double [][] output= new double[classification.size()][classification.size()];
        for (int a = 0; a <testingSet.length ; a++) {
            //tests all the testing set data
            String[] testingData = testingSet[a];

            double max = Double.MIN_VALUE;
            int indexOfMax = 0;

            for (int i = 0; i < classification.size(); i++) {
                double pifrom1tod = 1;
                for (int j = 0; j < values[0].length; j++) {
                    pifrom1tod *= F(values, classification, classification.get(i),j, Integer.parseInt(testingData[j]));
                }
                double potMax = 0;
                potMax = QofX(values, classification, classification.get(i)) * pifrom1tod;

                if (potMax > max) {
                    max = potMax;
                    indexOfMax = i;
                }
            }
            if(classification.get(indexOfMax).equals(testingData[testingData.length-1])){

                //gathers True Positives
                output[indexOfMax][indexOfMax]+=1;
            }
            else{

                //gathers false negatives, false positives
                for (int i = 0; i <classification.size() ; i++) {
                    if(classification.get(i).equals(testingData[testingData.length-1])){
                        output[indexOfMax][i]+=1;
                    }
                }
            }
        }
        return output;
    }

    /////////////////////////////////////////////////////////////
    /////function for F(Aj=ak, C=ci)
    private static double F(int [][][] values, ArrayList<String> classifications, String c,int attributeNum, int attributeValue){
        double output=1;
        double Nci=0;
        for (int i = 0; i <classifications.size() ; i++) {
            for (int j = 0; j < values[i][0].length; j++) {
                if(c.equals(classifications.get(i))){
                    Nci+=values[i][0][j];
                }
            }
        }
        for (int i = 0; i <values.length ; i++) {
            if(c.equals(classifications.get(i))){
                output=(values[i][attributeNum][attributeValue]+1)/(double)((values[i][attributeNum].length+Nci ));
            }
        }
        return output;
    }

    //////////////////////////////////////////////////////////////////
    /////function of Q(x)
    private static double QofX(int [][][] trainedData, ArrayList<String>classifications, String c){
        double currentClassValue=0;
        double total=0;
        for (int i = 0; i < classifications.size() ; i++) {

            for (int j = 0; j < trainedData[i][0].length; j++) {
                if(c.equals(classifications.get(i))){
                    currentClassValue+=trainedData[i][0][j];
                }
                total+=trainedData[i][0][j];
            }
        }
        return currentClassValue/total;
    }

    ///////////////////////////////////////////////////////////////
    //function for input data to be changed into a 2d string array
    private static String [][] fileTo2dString(File inputFile) throws Exception{

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

    /////////////////////////////////////////////////////////////////////////////////////
    //this function takes a 2d string array, and the specific data type it is processing
    //it then removes irrelevant data like id numbers, replaces missing data values
    //in the data sets, ensures that the classfications are the last column in the 2d matrix
    //and returns a 2d string array of the results to be further used by the Naive Bayes
    //algorithm.
    private static String[][] preprocessData(String [][] inputData, String dataType){
        //array for storing output
        String [][]outputData = new String [inputData.length][inputData[0].length];

        //conditional for processing glass data
        if(dataType.equals("glass")){
            //preprocessing glass data, removing id's because they are irrelevant
            //outputData is resized for that reason
            outputData = new String [inputData.length][inputData[0].length-1];
            for (int i = 0; i < inputData.length; i++) {
                for (int j = 1; j < inputData[0].length; j++) {
                    outputData[i][j-1]=inputData[i][j];
                }
            }
            //scale attribute data to integer values between 0-10 to make them discrete
            outputData=scaleValues(outputData,10);

        }
        //conditional for processing iris data
        else if (dataType.equals("iris")){
            for (int i = 0; i < inputData.length; i++) {
                for (int j = 0; j < inputData[0].length; j++) {
                    outputData[i][j]=inputData[i][j];
                }
            }
            //scale attribute data to integer values between 0-10 to make them discrete
            outputData=scaleValues(outputData,10);
        }
        //conditional for processing voter data
        else if (dataType.equals("votes")){
            //procedure for processing the vote data
            for (int i = 0; i < inputData.length; i++) {
                for (int j = 0; j < inputData[0].length; j++) {
                    //turns yes no abstain data into numerals
                    if(inputData[i][j].equals("?")){
                        outputData[i][j]="0";
                    }else if (inputData[i][j].equals("y")){
                        outputData[i][j]="1";
                    }else if (inputData[i][j].equals("n")){
                        outputData[i][j]="2";
                    }else{
                        outputData[i][j]=inputData[i][j];
                    }
                }
            }
            //swaps first and last column to be consistent with other data
            for (int i = 0; i <inputData.length ; i++) {
                String temp=outputData[i][0];
                outputData[i][0]=outputData[i][outputData[0].length-1];
                outputData[i][outputData[0].length-1]=temp;
            }
        }
        //conditional for processing cancer data
        else if (dataType.equals("cancer")){
            //process for dealing with missing data in the cancer dataset
            //outputData is one column less to trim ID numbers that have no relevance
            //so outputData is resized here
            outputData = new String [inputData.length][inputData[0].length-1];

            for (int i = 0; i < inputData.length; i++) {
                for (int j = 1; j < inputData[0].length; j++) {
                    if(inputData[i][j].equals("?")){
                        outputData[i][j-1]="1";//this 5 is arbitrary at the moment
                    }else{
                        outputData[i][j-1]=inputData[i][j];
                    }
                }
            }
        }
        //conditional for processing soybean data
        else if (dataType.equals("beans")){
            //no missing data in beans
            for (int i = 0; i < inputData.length; i++) {
                for (int j = 0; j < inputData[0].length; j++) {
                    outputData[i][j]=inputData[i][j];
                }
            }
        }
        return outputData;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    //function to return trained data information
    private static int[][][] trainingAlgorithm(String [][] inputData, ArrayList <String> classifications){

        //3d array to sorted by [classification][attribute][attributeValue] to hold counts
        //of attributes by attribute classification and value
        int maxNumAttributes=11;
        int[][][] attributeCount= new int[classifications.size()][inputData[0].length-1][maxNumAttributes];//currently 11 is hardcoded in
        //organizes all data into 3d array with counts by class number, attribute number, attribute value
        for (int i = 0; i <inputData.length ; i++) {
            for (int j = 0; j <inputData[0].length-1 ; j++) {
                for (int k = 0; k <classifications.size() ; k++) {
                    if(inputData[i][inputData[0].length-1].equals(classifications.get(k))){
                        attributeCount[k][j][Integer.parseInt(inputData[i][j])]++;
                    }
                }
            }
        }
        return attributeCount;
    }


    /////////////////////////////////////////////////////////////////////////////////
    //function to return array of unique classes in datasets
    private static ArrayList<String> getClassifications(String [][]inputData){
        //uses the last column of the array to make a list
        //of classifications
        ArrayList <String> classifications = new ArrayList<>();
        for (int i = 0; i <inputData.length ; i++) {
            if(classifications.contains(inputData[i][inputData[0].length-1])){

            }else{
                classifications.add(inputData[i][inputData[0].length-1]);
            }
        }
        return classifications;
    }

    /////////////////////////////////////////////////////////////////////////
    ///function to scale values of continuous variables
    private static String [][] scaleValues(String [][] outputData, int max){
        ///////////////////Scaling Section///////////////
        //array for holding max and mins of each attribute
        Double [][] maxAndMins= new Double [outputData[0].length-1][2];
        //filling maxes and mins with 0 and max int
        for (int i = 0; i <maxAndMins.length ; i++) {
            maxAndMins[i][0]=Double.MIN_VALUE;
            maxAndMins[i][1]=Double.MAX_VALUE;
        }
        //changing class data to a range of 10 possible values
        //finding max and mins
        for (int i = 0; i <outputData.length ; i++) {
            for (int j = 0; j <outputData[0].length-1 ; j++) {
                //max finding
                if(Double.parseDouble(outputData[i][j])>maxAndMins[j][0]){
                    maxAndMins[j][0]=Double.parseDouble(outputData[i][j]);
                }
                //min finding
                if(Double.parseDouble(outputData[i][j])<=maxAndMins[j][1]){
                    maxAndMins[j][1]=Double.parseDouble(outputData[i][j]);
                }
            }
        }
        //scale values to range from 0-max as integers
        for (int i = 0; i <outputData.length ; i++) {
            for (int j = 0; j <outputData[0].length-1; j++) {
                Double oldNum = Double.parseDouble(outputData[i][j]);
                //equation to scale numbers
                Double newNum=max*((oldNum-maxAndMins[j][1])/(maxAndMins[j][0]-maxAndMins[j][1]));
                outputData[i][j]= (int)Math.round(newNum)+"";
            }
        }
        return outputData;
    }
}
