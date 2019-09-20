import java.io.File;

public class Driver {

    public static void main(String args[])throws Exception{

        //for testing purposes
        Data d1 = new Data(new File("../DataSets/wine.data"),"wine");
        for (int k = 0; k <2 ; k++) {
            if(k%2==0){
                System.out.println("Training Set");
            }else{
                System.out.println("Testing Set");
            }
            for (int i = 0; i <d1.splitData.get(k).size() ; i++) {
                for (int j = 0; j <d1.splitData.get(k).get(0).size() ; j++) {
                    System.out.print(d1.splitData.get(k).get(i).get(j)+ " , ");
                }
                System.out.println();
            }


        }

    }
}
