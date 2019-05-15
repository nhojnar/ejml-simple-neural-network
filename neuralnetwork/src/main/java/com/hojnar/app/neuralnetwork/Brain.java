package com.hojnar.app.neuralnetwork;
import java.util.Random;

import org.ejml.simple.*;

public class Brain
{
	
	SimpleMatrix inputToHidden, hiddenToOutput;
	SimpleMatrix inputLayer, hiddenLayer, outputLayer;
	int inputNodes, hiddenNodes, outputNodes;
	
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
		inputToHidden = SimpleMatrix.random_DDRM(inputNodes, hiddenNodes, 0, 1, rand);
		hiddenToOutput = SimpleMatrix.random_DDRM(hiddenNodes, outputNodes, 0, 1, rand);
		
		inputLayer = new SimpleMatrix(1, inputNodes);
		hiddenLayer = new SimpleMatrix(1, hiddenNodes);
		outputLayer = new SimpleMatrix(1, outputNodes);
	}
	
	public double[] predict(double[] input)
	{
		inputLayer = new SimpleMatrix(1, inputNodes, true, input);
		
		hiddenLayer = inputLayer.mult(inputToHidden);
		sigmoid(hiddenLayer);
		
		outputLayer = hiddenLayer.mult(hiddenToOutput);
		sigmoid(outputLayer);
		
		double[] output = new double[outputNodes];
		for(int i = 0; i < outputNodes; i++)
			output[i] = outputLayer.get(i);
		return output;
	}
	
	double mut(double val, double rate)
	{
		if(Math.random() < rate)
		{
			return val + sigmoid(Math.random()) * .1;
		}
		return val;
	}
	void mutate(double rate)
	{
		for(int i = 0; i < inputToHidden.numRows(); i++)
		{
			for(int j = 0; j < inputToHidden.numCols(); j++)
			{
				double val = inputToHidden.get(i, j);
				inputToHidden.set(i, j, mut(val, rate));
			}
		}
		for(int i = 0; i < hiddenToOutput.numRows(); i++)
		{
			for(int j = 0; j < hiddenToOutput.numCols(); j++)
			{
				double val = hiddenToOutput.get(i, j);
				hiddenToOutput.set(i, j, mut(val, rate));
			}
		}
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
	double dtanh(double y)
	{
		return 1 / (Math.pow(Math.cosh(y), 2));
	}
}
