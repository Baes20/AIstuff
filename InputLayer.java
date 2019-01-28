package artificialintelligence;

public class InputLayer extends Layer{
	public InputLayer(Matrix input) {
		super(input.getHeight(), input.getWidth(), "");
		this.setNodes(input);
	}
}
