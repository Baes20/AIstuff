package artificialintelligence;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class MnistEpoch {
	public ArrayList<MnistPair> MnistList;
	
	public MnistEpoch() {
		MnistReader reader = new MnistReader();
		int[] labels = reader.getLabels("/Users/siwoobae/Downloads/train-labels-idx1-ubyte");
		List<int[][]> images = reader.getImages("/Users/siwoobae/Downloads/train-images-idx3-ubyte");
		
		
		MnistList = new ArrayList<MnistPair>();
		for(int i = 0; i < labels.length; i++) {
			MnistList.add(new MnistPair(linearize(images.get(i)), labels[i]));
		}
	}
	
	private double[] linearize(int[][] arr) {
		double[] Sample = new double[arr.length*arr[0].length];
		for(int i = 0; i < arr.length; i++) {
			for(int j = 0; j < arr[0].length; j++) {
				Sample[arr[0].length*i+j] = arr[i][j];
			}
		}
		return Sample;
	}
	
	public Stack<MnistPair> createMiniBatch(int paircount){
		Stack<MnistPair> temp = new Stack<MnistPair>();
		if(MnistList.size()>paircount) {
			for(int i = 0; i < paircount; i++) {
				temp.push(MnistList.remove((int)(Math.random()*MnistList.size())));
			}
		}else {
			while(!MnistList.isEmpty()) {
				temp.push(MnistList.remove((int)(Math.random()*MnistList.size())));
			}
		}
		return temp;
	}
	
	public boolean isEmpty() {
		return MnistList.isEmpty();
	}
	
	
}
