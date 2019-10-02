import java.io.File;

public class CarData extends Data {

    CarData(File inputFileName) throws Exception{
        fileTo2dStringArrayList(inputFileName);
        bucketize();
    }
}
