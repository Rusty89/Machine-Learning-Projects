/* A class that holds a CVS object. CVS stands for Cross Validation Set. This is a helper object that holds
    training sets, validation sets, and test sets for each instance of cross validation, in our case 10.
 */

import java.util.ArrayList;

// defines a Cross Validation Set as a group of training sets, validation sets, and test sets
public class CVS {

    public ArrayList<ArrayList<ArrayList<String>>> trainingSets = new ArrayList<>();
    public ArrayList<ArrayList<ArrayList<String>>> validationSets  = new ArrayList<>();
    public ArrayList<ArrayList<ArrayList<String>>> testSets = new ArrayList<>();

}