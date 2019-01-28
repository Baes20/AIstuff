package artificialintelligence;

import java.util.ArrayList;

public class MLP {
	private ArrayList<Layer> Layers;
	private Matrix currentInput;
	private Matrix currentAnswer;
	private int currentLabel;
	
	public MLP(Matrix InputMatrix) {
		Layers = new ArrayList<Layer>();
		Layers.add(new InputLayer(InputMatrix));
	}
	
	public void add(int height, String act) {
		Layer1D l = new Layer1D(height,act,Layers.get(Layers.size()-1));
		Layers.add(l);
		Layers.get(Layers.size()-2).nextLayer = Layers.get(Layers.size()-1);
	}
	
	public Layer get(int index) {
		return Layers.get(index);
	}
	
	public void randomizeAllWeights() {
		for(int i = 0; i < Layers.size(); i++) {
			Layer currentLayer = Layers.get(i);
			if(currentLayer instanceof Layer1D) {
				((Layer1D)currentLayer).randomize();
			}
		}
	}
	
	public void setCurrentInput(Matrix in) {
		currentInput = in;
	}
	
	public void setCurrentAnswer(Matrix ans) {
		currentAnswer = ans;
	}
	
	public void setCurrentLabel(int l) {
		currentLabel = l;
	}
	
	public Matrix feedforward() {
		Layers.get(0).setNodes(currentInput);
		for(int i = 1; i < Layers.size(); i++) {
			((Layer1D)Layers.get(i)).update();
		}
		return Layers.get(Layers.size()-1).getNodes();
	}
	
	public void backpropagate() {
		feedforward();
		Layer aL = Layers.get(Layers.size()-1);
		Matrix a = Matrix.Hadamard(Matrix.Subtract(aL.nodes, currentAnswer), aL.dActivationFunction.activate(aL.z));
		//System.out.println(a);
		((Layer1D)aL).setError(a);
		for(int i = Layers.size()-2; i >= 1; i--) {
			((Layer1D)Layers.get(i)).backpropagateError();
		}
		for(int i = Layers.size()-1; i >= 1; i--) {
			((Layer1D)Layers.get(i)).accumulateTodWdB();
		}
	}
	
	public void applyGradientDescent(double learningRate, int miniBatchSize) {
		for(int i = Layers.size()-1; i >= 1; i--) {
			((Layer1D)Layers.get(i)).descend(learningRate, miniBatchSize);
		}
		for(int i = Layers.size()-1; i >= 1; i--) {
			((Layer1D)Layers.get(i)).reset();
		}
	}
	
	public int size() {
		return Layers.size();
	}
	
	public double getCost() {
		Matrix costV = Matrix.Subtract(currentAnswer, feedforward());
		double sum = 0;
		for(int i = 0; i < costV.getHeight(); i++) {
			sum += costV.get(i, 0)*costV.get(i, 0);
		}
		return sum/2.0;
	}
	
	public int guess() {
		Matrix costV = feedforward();
		double Max = Double.MIN_VALUE;
		int indexOfMax = -1;
		for(int i = 0; i < costV.getHeight(); i++) {
			if( Max < costV.get(i, 0)) {
				Max = costV.get(i, 0);
				indexOfMax = i;
			}
		}
		if(indexOfMax == currentLabel) {
			return 1;
		}else {
			return 0;
		}
	}
	
	
	
	
}
