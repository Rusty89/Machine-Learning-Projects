import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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


    public static String accuracy(ArrayList<String> results, ArrayList<ArrayList<String>> testData){

        double truePos=0;
        double total = 0;
        for (int i = 0; i <testData.size() ; i++) {
            String guess = results.get(i);
            String actual = testData.get(i).get(testData.get(0).size()-1);

            if(guess.equals(actual)){
                truePos++;
                total++;
            }else{
                total++;
            }

        }

        return (truePos/total)+"";
    }




}
