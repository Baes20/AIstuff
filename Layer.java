package artificialintelligence;

public class Layer {
	protected Matrix nodes;
	protected Matrix biases;
	protected Matrix z;
	public Activation activationFunction;
	public Activation dActivationFunction;
	public Layer prevLayer;
	public Layer nextLayer;
	
	public Layer(int height, int width, String act) {
		nodes = new Matrix(height, width);
		biases = new Matrix(height, width);
		z = new Matrix(height, width);
		prevLayer = null;
		nextLayer = null;
		if(act.trim().toLowerCase().equals("relu")) {
			activationFunction = (mat) -> Matrix.ReLU(mat);
			dActivationFunction = (mat) -> Matrix.dReLU(mat);
		}else if(act.trim().toLowerCase().equals("sigmoid")) {
			activationFunction = (mat) -> Matrix.Sigmoid(mat);
			dActivationFunction = (mat) -> Matrix.dSigmoid(mat);
		}
	}
	
	public Matrix getNodes() {
		return nodes;
	}
	
	public void setNodes(Matrix a) {
		if(a.getHeight() == nodes.getHeight() && a.getWidth() == nodes.getWidth()) {
			nodes = a;
		}else {
			throw new ArrayIndexOutOfBoundsException();
		}
	}
	
	public Matrix getBiases() {
		return biases;
	}
	
	public void setBiases(Matrix a) {
		if(a.getHeight() == biases.getHeight() && a.getWidth() == biases.getWidth()) {
			biases = a;
		}else {
			throw new ArrayIndexOutOfBoundsException();
		}
	}
	
	public Matrix getZ() {
		return z;
	}
	
	public void setZ(Matrix a) {
		if(a.getHeight() == z.getHeight() && a.getWidth() == z.getWidth()) {
			z = a;
		}else {
			throw new ArrayIndexOutOfBoundsException();
		}
	}
}
