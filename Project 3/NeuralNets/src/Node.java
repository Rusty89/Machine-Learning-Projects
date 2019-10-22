import java.util.ArrayList;

public class Node
{
    private ArrayList<Double> vector;
    private double weight;

    public Node (ArrayList<Double> vector)
    {
        this.vector = vector;
    }
    public Node (double weight)
    {
        this.weight = weight;
    }
}
