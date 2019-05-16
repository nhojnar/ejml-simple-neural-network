package com.hojnar.app.neuralnetwork;

import java.util.Random;
import org.ejml.simple.SimpleMatrix;

public abstract class Brain 
{
	SimpleMatrix inputToHidden, hiddenToOutput;
	SimpleMatrix inputLayer, hiddenLayer, outputLayer;
	public int inputNodes, hiddenNodes, outputNodes;
	
	public Brain(Brain copy)
	{
		inputNodes = copy.inputNodes;
		hiddenNodes = copy.hiddenNodes;
		outputNodes = copy.outputNodes;
		inputToHidden = copy.inputToHidden.copy();
		hiddenToOutput = copy.hiddenToOutput.copy();
		
		inputLayer = new SimpleMatrix(1, inputNodes);
		hiddenLayer = new SimpleMatrix(1, hiddenNodes);
		outputLayer = new SimpleMatrix(1, outputNodes);
	}
	public Brain(int inputNodes, int hiddenNodes, int outputNodes)
	{
		this.inputNodes = inputNodes;
		this.hiddenNodes = hiddenNodes;
		this.outputNodes = outputNodes;
		
		Random rand = new Random();
		inputToHidden = SimpleMatrix.random_DDRM(inputNodes, hiddenNodes, -1, 1, rand);
		hiddenToOutput = SimpleMatrix.random_DDRM(hiddenNodes, outputNodes, -1, 1, rand);
		
		inputLayer = new SimpleMatrix(1, inputNodes);
		hiddenLayer = new SimpleMatrix(1, hiddenNodes);
		outputLayer = new SimpleMatrix(1, outputNodes);
	}
	
	public abstract double[] predict(double[] input);
	
	public SimpleMatrix getFirstMatrix()
	{
		return inputToHidden.copy();
	}
	public SimpleMatrix getSecondMatrix()
	{
		return hiddenToOutput.copy();
	}
	
	double sigmoid(double x)
	{
		return 1 / (1+Math.exp(-x));
	}
	void sigmoid(SimpleMatrix a)
	{
		for(int i = 0; i < a.numRows(); i++)
		{
			for(int j = 0; j < a.numCols(); j++)
			{
				double val = a.get(i, j);
				a.set(i, j, sigmoid(val));
			}
		}
	}
	double dsigmoid(double y)
	{
		return y * (1-y);
	}
	double tanh(double x)
	{
		return Math.tanh(x);
	}
	void tanh(SimpleMatrix a)
	{
		for(int i = 0; i < a.numRows(); i++)
		{
			for(int j = 0; j < a.numCols(); j++)
			{
				double val = a.get(i,j);
				a.set(i, j, tanh(val));
			}
		}
	}
	double dtanh(double y)
	{
		return 1 / (Math.pow(Math.cosh(y), 2));
	}
	
	
	public void exportBrain(String name)
	{
		SimpleMatrix export = new SimpleMatrix(hiddenNodes + inputNodes + 1, (hiddenNodes > outputNodes) ? hiddenNodes : outputNodes);
		
		export.set(0, 0, inputNodes);
		export.set(0, 1, hiddenNodes);
		export.set(0, 2, outputNodes);
		
		for(int i = 0; i < inputToHidden.numRows(); i++)
		{
			for(int j = 0; j < inputToHidden.numCols(); j++)
			{
				export.set(i+1, j, inputToHidden.get(i, j));
			}
		}
		
		for(int i = 0; i < hiddenToOutput.numRows(); i++)
		{
			for(int j = 0; j < hiddenToOutput.numCols(); j++)
			{
				export.set(inputToHidden.numRows()+i+1, j, hiddenToOutput.get(i, j));
			}
		}
		
		try {
		export.saveToFileCSV(name+".brain");
		} catch(Exception e)
		{
			e.printStackTrace();
			return;
		}
		System.out.printf("Exported brain as %s.brain\n", name);
	}
	
	public void importBrain(String name)
	{
		SimpleMatrix temp = new SimpleMatrix(0,0);
		SimpleMatrix imp;
		try {
			imp = temp.loadCSV(name+".brain");
		}  catch(Exception e)
		{
			e.printStackTrace();
			return;
		}
		inputNodes = (int)imp.get(0, 0);
		hiddenNodes = (int)imp.get(0,1);
		outputNodes = (int)imp.get(0,2);
		
		inputToHidden = new SimpleMatrix(inputNodes, hiddenNodes);
		for(int i = 0; i < inputToHidden.numRows(); i++)
		{
			for(int j = 0; j < inputToHidden.numCols(); j++)
			{
				inputToHidden.set(i, j, imp.get(i+1, j));
			}
		}
		hiddenToOutput = new SimpleMatrix(hiddenNodes, outputNodes);
		for(int i = 0; i < hiddenToOutput.numRows(); i++)
		{
			for(int j = 0; j < hiddenToOutput.numCols(); j++)
			{
				hiddenToOutput.set(i, j, imp.get(inputToHidden.numCols()+i, j));
			}
		}
		System.out.printf("Imported %s.brain", name);
	}
	
	public void print()
	{
		System.out.println("Input to Hidden");
		inputToHidden.print();
		System.out.println("Hidden to Output");
		hiddenToOutput.print();
	}
	
}
