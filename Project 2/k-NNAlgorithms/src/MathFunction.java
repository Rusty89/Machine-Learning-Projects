import java.util.*;

public class MathFunction {

    public static String euclideanDistance(List<String> X1Vector, List<String> X2Vector){
        double result = 0;
        int sizeOfX1andX2 = X1Vector.size();
        for (int i = 0; i <sizeOfX1andX2 ; i++) {
            //sqrt not used as it won't effect the result, but will cost computing time
            double x = Double.parseDouble(X1Vector.get(i));
            double y = Double.parseDouble(X2Vector.get(i));
            //summation of (x1i-x2i)^2, euclidean distance
            result += Math.pow((x-y), 2);
        }
        return result+"";
    }


    public static String hammingDistance(List<String> in1, List<String> in2){
        double result = 0;
        for (int i = 0; i <in1.size() ; i++) {
            if(in1.get(i).equals(in2.get(i))){
                //do not increment result, hamming dist 0 between these two points
            }else{
                result++;
            }
        }
        return result+"";
    }

    public static String average (ArrayList<String> input){
        double sum=0;
        for (int i = 0; i <input.size() ; i++) {
            sum+= Double.parseDouble(input.get(i));
        }
        sum/=input.size();
        return sum+"";
    }

    public static String mode(ArrayList<String> input){
        String mode="";
        int maxIndex=0;
        int currMax=0;
        for (int i = 0; i <input.size() ; i++) {
            int countOfClass = Collections.frequency(input,input.get(i));
            if(countOfClass>=currMax){
                maxIndex=i;
            }
        }
        mode=input.get(maxIndex);
        return mode;
    }


    public static ArrayList<String> processConfusionMatrix(ArrayList<String> results, ArrayList<ArrayList<String>> testData){

        int lengthOfData = results.size();
        int classIndex = testData.get(0).size()-1;

        //initialize classifications in the confusion matrix as the first row
        ArrayList<ArrayList<String>> confusionMatrix= new ArrayList<>();
        confusionMatrix.add(new ArrayList<>());
        for (int i = 0; i <lengthOfData ; i++) {
            if(!confusionMatrix.get(0).contains(testData.get(i).get(classIndex))){
                confusionMatrix.get(0).add(testData.get(i).get(classIndex));
            }
            if(!confusionMatrix.get(0).contains(results.get(i))){
                confusionMatrix.get(0).add(results.get(i));
            }
        }
        //initialize the rest of the matrix as an n by n matrix below the classification row
        //filled with 0 initially
        for (int i = 0; i <confusionMatrix.get(0).size(); i++) {
            confusionMatrix.add(new ArrayList<>());
            for (int j = 0; j <confusionMatrix.get(0).size() ; j++) {
                confusionMatrix.get(i+1).add("0");
            }
        }

        //populate the matrix by indexing guesses by their
        //true value and the guess
        for (int i = 0; i <lengthOfData; i++) {
            String guess = results.get(i);
            String actual = testData.get(i).get(testData.get(0).size()-1);
            int indexOfActual = confusionMatrix.get(0).indexOf(actual);
            int indexOfGuess = confusionMatrix.get(0).indexOf(guess);
            int currentValAtPos= Integer.parseInt(confusionMatrix.get(indexOfActual+1).get(indexOfGuess));
            currentValAtPos++;
            confusionMatrix.get(indexOfActual+1).set(indexOfGuess, currentValAtPos+"");
        }

        //calculate truePos, falsePos, falseNeg and totalPos by indexing the confusion
        //matrix appropriately
        double truePos=0;
        double totalPos=0;
        double falsePos=0;
        double falseNeg=0;
        double precisionSum=0;
        double recallSum=0;
        for (int i = 1; i <confusionMatrix.size() ; i++) {
            totalPos+=Integer.parseInt(confusionMatrix.get(i).get(i-1));
            truePos=Integer.parseInt(confusionMatrix.get(i).get(i-1));
            for (int j = 0; j <confusionMatrix.get(0).size() ; j++) {

                falseNeg+=Integer.parseInt(confusionMatrix.get(i).get(j));
                falsePos+=Integer.parseInt(confusionMatrix.get(j+1).get(i-1));
            }

            falseNeg-=truePos;//get rid of true pos value in these results
            falsePos-=truePos;

            //calculates the precision and recall for each class
            //adds it to the summation to later be divided by
            //the number of classes
            //if statements to protect against 0/0 situations
            if(!Double.isNaN(truePos/truePos+falsePos)){
                precisionSum+= truePos/(truePos+falsePos);
            }
            if(!Double.isNaN(truePos/truePos+falseNeg)){
                recallSum+= truePos/(truePos+falseNeg);
            }



        }
        //divides sums by number of classes to get
        //overall precision and recall
        precisionSum/=confusionMatrix.get(0).size();
        recallSum/=confusionMatrix.get(0).size();
        //calculate overall accuracy by dividing all
        //truePos results and dividing by all guessed
        //data points.
        double accuracy = totalPos/results.size();
        ArrayList<String> result = new ArrayList<String>();
        result.add(precisionSum+"");
        result.add(recallSum+"");
        result.add(accuracy+"");
        return result;
    }


    public static String rootMeanSquaredError(ArrayList<String> results, ArrayList<ArrayList<String>> testData,  ArrayList<ArrayList<String>> fullSet){
        double sum=0;
        double max=Double.MIN_VALUE;
        double min=Double.MAX_VALUE;

        //get max and min values in the set to normalize results
        for (int i = 0; i <fullSet.size() ; i++) {
            if(max<=Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size()-1))){
                max=Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size()-1));
            }
            if(min>=Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size()-1))){
                min=Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size()-1));
            }
        }
        for (int i = 0; i <testData.size() ; i++) {
            double guess = Double.parseDouble(results.get(i));
            double actual = Double.parseDouble(testData.get(i).get(testData.get(0).size()-1));

            sum+=Math.pow((guess-actual),2);

        }
        //average the summation
        sum/=testData.size();
        sum=Math.sqrt(sum);
        //normalize to a percentage
        sum/=(max-min);
        return sum+"";
    }

    public static String meanAbsoluteError(ArrayList<String> results, ArrayList<ArrayList<String>> testData,ArrayList<ArrayList<String>> fullSet){
        double sum=0;
        double max=Double.MIN_VALUE;
        double min=Double.MAX_VALUE;

        //get max and min values in the set to normalize results
        for (int i = 0; i <fullSet.size() ; i++) {
            if(max<=Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size()-1))){
                max=Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size()-1));
            }
            if(min>=Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size()-1))){
                min=Double.parseDouble(fullSet.get(i).get(fullSet.get(0).size()-1));
            }
        }
        for (int i = 0; i <testData.size() ; i++) {
            double guess = Double.parseDouble(results.get(i));
            double actual = Double.parseDouble(testData.get(i).get(testData.get(0).size()-1));

            sum+=Math.abs(guess-actual);

        }
        //average the summation
        sum/=testData.size();

        //normalize to a percentage
        sum/=(max-min);
        return sum+"";
    }



}
