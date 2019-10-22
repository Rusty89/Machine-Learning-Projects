/* A class that holds a Cluster object. Clusters are groups of points that define a region in a data set that
    share a classification. They are used in the K-Means and PAM algorithms.
 */

import java.util.ArrayList;

// clusters are an objects used to group data points in the K-Means and PAM algorithms
public class Cluster
{
    private ArrayList<String> medoid, previusMedoid;  // only allowed to update through the setter
    ArrayList<ArrayList<String>> points;

    public Cluster (ArrayList<String> medoid)
    {
        this.medoid = medoid;
        previusMedoid = null;
        points = new ArrayList<>();
        points.add(medoid);
    }

    public ArrayList<String> getAverage()
    {
        // initialize multiple ArrayLists to send to MathFunction.average
        ArrayList<ArrayList<String>> values = new ArrayList<>();
        for (int a = 0; a < medoid.size() - 1; a++) // skip over the classification at the end of the ArrayList
            values.add(new ArrayList<>());

        // add each point's value for every particular attribute to an ArrayList. one list per feature/attribute.
        for (ArrayList<String> point: points)
        {
            for (int a = 0; a < point.size() - 1; a++)
                values.get(a).add(point.get(a));
        }

        // calculate the average over each individual attribute
        ArrayList<String> average = new ArrayList<>();
        for (ArrayList<String> list: values)
            average.add(MathFunction.average(list));

        // return a new point at that location
        return average;
    }

    // automatically tracks the last medoid
    public void setMedoid (ArrayList<String> newMedoid)
    {
        previusMedoid = medoid;
        medoid = newMedoid;
    }

    public ArrayList<String> getMedoid()
    {
        return medoid;
    }
    public boolean medoidMoved()
    {
        return medoid != previusMedoid;
    }
}
