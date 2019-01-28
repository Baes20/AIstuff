package artificialintelligence;

import java.util.Arrays;

public class Matrix {
	private int height;
	private int width;
	private double[][] arr;
	
	public Matrix(int m, int n) { // Tested
		height = m;
		width = n;
		arr = new double[m][n];
	}
	
	public Matrix(double[][] input) {
		height = input.length;
		width = input[0].length;
		arr = input;
	}
	
	public Matrix(double[] Vector) {
		height = Vector.length;
		width = 1;
		arr = new double[height][width];
		for(int i = 0; i < arr.length; i++) {
			arr[i][0] = Vector[i];
		}
	}
	
	public void reset() {
		arr = new double[height][width];
	}
	
	public int getHeight() { // Tested
		return height;
	}
	
	public int getWidth() { // Tested
		return width;
	}
	
	public double get(int i, int j) { // Tested
		return arr[i][j];
	}
	
	public void set(int i, int j, double num) { // Tested
		arr[i][j] = num;
	}
	
	public String toString() { // Tested
		String ret = "";
		for(int i = 0; i < this.height; i++) {
			ret += Arrays.toString(arr[i])+"\n";
		}
		return ret;
	}
	
	public void randomize(double lower, double upper) { // Tested
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				arr[i][j] = ((upper-lower)*Math.random()+lower);
			}
		}
	}
	
	public static Matrix Add(Matrix a, Matrix b) { // Tested
		
		if( a.getWidth() == b.getWidth() && a.getHeight() == b.getHeight()) {
			Matrix ret = new Matrix(a.getHeight(), a.getWidth());
			for(int i = 0; i < ret.getHeight(); i++) {
				for(int j = 0; j < ret.getWidth(); j++) {
					double Anum = a.get(i, j);
					double Bnum = b.get(i, j);
					ret.set(i, j, Anum + Bnum );
				}
			}
			return ret;
		}else {
			throw new ArrayIndexOutOfBoundsException();
		}
	}
	
	public static Matrix Subtract(Matrix a, Matrix b) { // Tested
		if( a.getWidth() == b.getWidth() && a.getHeight() == b.getHeight()) {
			Matrix ret = new Matrix(a.getHeight(), a.getWidth());
			for(int i = 0; i < ret.getHeight(); i++) {
				for(int j = 0; j < ret.getWidth(); j++) {
					double Anum = a.get(i, j);
					double Bnum = b.get(i, j);
					ret.set(i, j, Anum - Bnum );
				}
			}
			return ret;
		}else {
			throw new ArrayIndexOutOfBoundsException();
		}
	}
	
	public static Matrix Mult(Matrix a, Matrix b) { // Tested
		if( a.width == b.height ) {
			Matrix ret = new Matrix(a.getHeight(), b.getWidth());
			for(int i = 0; i < ret.height; i++) {
				for(int j = 0; j < ret.width; j++) {
					double sum = 0;
					for(int k = 0; k < a.width; k++) {
						sum += a.arr[i][k] * b.arr[k][j];
					}
					ret.arr[i][j] = sum;
				}
			}
			return ret;
		}else {
			throw new ArrayIndexOutOfBoundsException();
		}
	}
	
	public static Matrix Mult(Matrix a, double C) {
		Matrix ret = new Matrix(a.getHeight(), a.getWidth());
		for(int i = 0; i < ret.height; i++) {
			for(int j = 0; j < ret.width; j++) {
				ret.arr[i][j] = C*a.arr[i][j];
			}
		}
		return ret;
	}
	
	public static Matrix T(Matrix a) { // Tested
		Matrix ret = new Matrix(a.width, a.height);
		for(int i = 0; i < ret.height; i++) {
			for(int j = 0; j < ret.width; j++) {
				ret.arr[i][j] = a.arr[j][i];
			}
		}
		return ret;
	}
	
	public static Matrix Hadamard(Matrix a, Matrix b) {
		if( a.getWidth() == b.getWidth() && a.getHeight() == b.getHeight()) {
			Matrix ret = new Matrix(a.getHeight(), a.getWidth());
			for(int i = 0; i < ret.getHeight(); i++) {
				for(int j = 0; j < ret.getWidth(); j++) {
					double Anum = a.get(i, j);
					double Bnum = b.get(i, j);
					ret.set(i, j, Anum * Bnum );
				}
			}
			return ret;
		}else {
			throw new ArrayIndexOutOfBoundsException();
		}
	}

	public void applySigmoid() {
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				arr[i][j] = Sigmoid(arr[i][j]);
			}
		}
	}
	
	public void applydSigmoid() {
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				arr[i][j] = dSigmoid(arr[i][j]);
			}
		}
	}
	
	public void applyReLU() {
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				arr[i][j] = ReLU(arr[i][j]);
			}
		}
	}
	
	public void applydReLU() {
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				arr[i][j] = dReLU(arr[i][j]);
			}
		}
	}
	
	public void applySoftMax() {
		for(int j = 0; j < width; j++) {
			double sum = 0;
			for(int i = 0; i < height; i++) {
				sum += Math.exp(arr[i][j]);
			}
			for(int i = 0; i < height; i++) {
				arr[i][j] = Math.exp(arr[i][j]) / sum;
			}
		}
	}
	
	private double KronDelta(int i, int j) {
		if(i == j) {
			return 1;
		}else {
			return 0;
		}
	}
	
	private static Matrix SoftMax(Matrix x) {
		Matrix ret = new Matrix(x.height, x.width);
		for(int j = 0; j < ret.width; j++) {
			double sum = 0;
			for(int i = 0; i < ret.height; i++) {
				sum += Math.exp(x.arr[i][j]);
			}
			for(int i = 0; i < ret.height; i++) {
				ret.arr[i][j] = Math.exp(x.arr[i][j]) / sum;
			}
		}
		return ret;
	}
	
	public static Matrix ReLU(Matrix a) {
		Matrix ret = new Matrix(a.height,a.width);
		for(int i = 0; i < ret.height; i++) {
			for(int j = 0; j < ret.width; j++) {
				ret.arr[i][j] = ReLU(a.arr[i][j]);
			}
		}
		return ret;
	}
	
	public static double ReLU(double x) {
		if(x < 0) {
			return 0;
		}else {
			return x;
		}
	}
	
	public static Matrix dReLU(Matrix a) {
		Matrix ret = new Matrix(a.height,a.width);
		for(int i = 0; i < ret.height; i++) {
			for(int j = 0; j < ret.width; j++) {
				ret.arr[i][j] = dReLU(a.arr[i][j]);
			}
		}
		return ret;
	}
	
	public static double dReLU(double x) {
		if(x < 0) {
			return 0;
		}else {
			return 1;
		}
	}
	

	public static double Sigmoid(double x) {
		return 1/(1+Math.pow(Math.E, -x));
	}
	
	public static Matrix Sigmoid(Matrix a) {
		Matrix ret = new Matrix(a.height,a.width);
		for(int i = 0; i < ret.height; i++) {
			for(int j = 0; j < ret.width; j++) {
				ret.arr[i][j] = Sigmoid(a.arr[i][j]);
			}
		}
		return ret;
	}
	
	public static double dSigmoid(double x) {
		return Sigmoid(x)*(1-Sigmoid(x));
	}
	
	public static Matrix dSigmoid(Matrix a) {
		Matrix ret = new Matrix(a.height,a.width);
		for(int i = 0; i < ret.height; i++) {
			for(int j = 0; j < ret.width; j++) {
				ret.arr[i][j] = dSigmoid(a.arr[i][j]);
			}
		}
		return ret;
	}

	
	
}
