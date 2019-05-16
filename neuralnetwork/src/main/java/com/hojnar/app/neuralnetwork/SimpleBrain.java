package com.hojnar.app.neuralnetwork;
import java.util.Random;

import org.ejml.simple.*;

public class SimpleBrain extends Brain
{
	
	public static void main(String[] args)
	{
		SimpleBrain b = new SimpleBrain(3, 5, 2);
		System.out.println(b);
		b.predict(new double[] {.97, -.3, .32});
		System.out.println("\n\n");
	}
	
	public SimpleBrain(SimpleBrain copy)
	{
		super(copy);
	}
	
	public SimpleBrain(int inputNodes, int hiddenNodes, int outputNodes)
	{
		super(inputNodes, hiddenNodes, outputNodes);
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
	

}
