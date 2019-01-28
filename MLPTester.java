package artificialintelligence;

import java.util.Stack;

public class MLPTester {
	
	public static void main(String[] args) {
		MnistEpoch trainingexamples = new MnistEpoch();
		int initialsize = trainingexamples.MnistList.size();
		double learningRate = 12;
		int miniBatchSize = 300;
		
		MLP Test = new MLP(new Matrix(784,1));
		Test.add(392,"ReLU");
		Test.add(196,"ReLU");
		Test.add(98, "ReLU");
		Test.add(49, "ReLU");
		Test.add(25, "ReLU");
		Test.add(10, "sigmoid");
		Test.randomizeAllWeights();
		for(int i = 0; i < 10; i++) {
			printAccuracy(Test);
			trainingexamples = new MnistEpoch();
			while(!trainingexamples.isEmpty()) {
				Stack<MnistPair> miniBatch = trainingexamples.createMiniBatch(miniBatchSize);
				while(!miniBatch.isEmpty()) {
					MnistPair currentExample = miniBatch.pop();
					Matrix currentImage = new Matrix(currentExample.getLinImage());
					Matrix currentAnswer = new Matrix(currentExample.getAnswer());
					Test.setCurrentInput(currentImage);
					Test.setCurrentAnswer(currentAnswer);
					Test.backpropagate();
				}
				Test.applyGradientDescent(learningRate, miniBatchSize);
			}
			
		}
	}
	
	public static void printAccuracy(MLP a) {
		MnistEpoch trainingexamples = new MnistEpoch();
		Stack<MnistPair> miniBatch = trainingexamples.createMiniBatch(5000);
		double totalCost = 0;
		double totalAccuracy = 0;
		while(!miniBatch.isEmpty()) {
			MnistPair currentExample = miniBatch.pop();
			Matrix currentImage = new Matrix(currentExample.getLinImage());
			Matrix currentAnswer = new Matrix(currentExample.getAnswer());
			a.setCurrentInput(currentImage);
			a.setCurrentAnswer(currentAnswer);
			a.setCurrentLabel(currentExample.getlabel());
			totalCost += a.getCost();
			totalAccuracy += a.guess();
		}
		double avgCost = totalCost/5000;
		double avgAccuracy = totalAccuracy/5000;
		
		System.out.println("average Cost is: "+avgCost+", Accuracy is: "+avgAccuracy);
		
	}
	
	
	
	
	
	
}
