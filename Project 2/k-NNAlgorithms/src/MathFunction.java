import java.util.ArrayList;

public class MathFunction {

    public static double euclideanDistance(ArrayList<String> X1Vector, ArrayList<String> X2Vector){
        double result = 0;
        int sizeOfX1andX2 = X1Vector.size();
        for (int i = 0; i <sizeOfX1andX2 ; i++) {
            //sqrt not used as it won't effect the result, but will cost computing time
            double x = Double.parseDouble(X1Vector.get(i));
            double y = Double.parseDouble(X2Vector.get(i));
            //summation of (x1i-x2i)^2, euclidean distance
            result += Math.pow((x-y), 2);
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
