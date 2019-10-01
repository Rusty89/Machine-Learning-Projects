import java.util.ArrayList;

public class MathFunction {

    public static double euclideanDistance(ArrayList<String> in1, ArrayList<String> in2){
        double result = 0;
        for (int i = 0; i <in1.size() ; i++) {
            //sqrt not used as it won't effect the result, but will cost computing time
            result += Math.pow(Double.parseDouble(in1.get(i))-Double.parseDouble(in2.get(i)), 2);
        }
        return result;
    }


    public static double hammingDistance(ArrayList<String> in1, ArrayList<String> in2){
        double result = 0;
        for (int i = 0; i <in1.size() ; i++) {
            if(in1.get(i).equals(in2.get(i))){
                //do not increment result, hamming dist 0 between these two points
            }else{
                result++;
            }
        }
        return result;
    }
}
