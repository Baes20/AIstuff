package artificialintelligence;

public class Layer1D extends Layer {
	
	private Matrix weights;
	private Matrix error;
	private Matrix dW;
	private Matrix dB;
	
	public Layer1D(int height, String act, Layer prev) {
		super(height, 1, act);
		error = new Matrix(height, 1);
		prevLayer = prev;
		weights = new Matrix(this.nodes.getHeight(), prevLayer.nodes.getHeight());
		dW = new Matrix(this.nodes.getHeight(), prevLayer.nodes.getHeight());
		dB = new Matrix(nodes.getHeight(), nodes.getWidth());
	}
	
	public void randomize() {
		double d = Math.sqrt(prevLayer.nodes.getHeight()*prevLayer.nodes.getWidth());
		weights.randomize(-1/d, 1/d);
	
	}
	
	public void update() {
		Matrix temp = Matrix.Mult(weights, prevLayer.nodes);
		z = Matrix.Add(temp, biases);
		nodes = Matrix.Sigmoid(z);
		//System.out.println(prevLayer.nodes);
		//System.out.println(nodes);
	}
	
	public void backpropagateError() {
		if(!(nextLayer == null)) {
			Layer1D next = ((Layer1D)nextLayer);
			Matrix wT = Matrix.T(next.weights);
			error = Matrix.Hadamard(Matrix.Mult(wT, next.error), this.dActivationFunction.activate(z));
		}
	}
	
	public void reset() {
		error.reset();
		nodes.reset();
		z.reset();
		dB.reset();
		dW.reset();
	}
	
	public void accumulateTodWdB() {
		Matrix temp = Matrix.Mult(error, Matrix.T(prevLayer.getNodes()));
		dB = Matrix.Add(dB, error);
		dW = Matrix.Add(dW, temp);
	}
	
	public void descend(double learningRate, double MinibatchSize) {
		weights = Matrix.Subtract(weights, Matrix.Mult(dW, learningRate/MinibatchSize));
		biases =  Matrix.Subtract(biases, Matrix.Mult(dB, learningRate/MinibatchSize));
	}
	
	public void setError(Matrix a) {
		if(a.getHeight() == nodes.getHeight() && a.getWidth() == nodes.getWidth()) {
			error = a;
		}else {
			throw new ArrayIndexOutOfBoundsException();
		}
	}
	
	public Matrix getError() {
		return error;
	}
	
	
}
