package artificialintelligence;

import java.util.List;

public class MnistPair {
	private int label;
	private double[] answer;
	private double[] linearizedimage;
	
	public MnistPair(double[] linimg, int label){
		linearizedimage = linimg;
		answer = new double[10];
		answer[label] = 1;
		this.label = label;
	}
	
	public double[] getAnswer() {
		return answer;
	}
	
	public double[] getLinImage(){
		return linearizedimage;
	}
	
	public int getlabel() {
		return label;
	}
	
}
	
	


