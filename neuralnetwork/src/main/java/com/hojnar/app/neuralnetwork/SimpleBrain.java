package com.hojnar.app.neuralnetwork;
import org.ejml.simple.*;
import java.util.Random;

public class SimpleBrain extends Brain
{
	private SimpleBrain()
	{
		super(0,0,0);
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
		tanh(hiddenLayer);
		
		outputLayer = hiddenLayer.mult(hiddenToOutput);
		sigmoid(outputLayer);
		
		double[] output = new double[outputNodes];
		for(int i = 0; i < outputNodes; i++)
			output[i] = outputLayer.get(i);
		return output;
	}
	

	double mut(double val, double rate, double intensity)
	{
		Random rand = new Random();
		if(rand.nextDouble() < rate)
		{
			return tanh(val + rand.nextGaussian() * intensity);
		}
		return val;
	}
	public void mutate(double rate, double intensity)
	{
		for(int i = 0; i < inputToHidden.numRows(); i++)
		{
			for(int j = 0; j < inputToHidden.numCols(); j++)
			{
				double val = inputToHidden.get(i, j);
				inputToHidden.set(i, j, mut(val, rate, intensity));
			}
		}
		for(int i = 0; i < hiddenToOutput.numRows(); i++)
		{
			for(int j = 0; j < hiddenToOutput.numCols(); j++)
			{
				double val = hiddenToOutput.get(i, j);
				hiddenToOutput.set(i, j, mut(val, rate, intensity));
			}
		}
	}
	public void mutate(double rate)
	{
		mutate(rate, 0.15);
	}
	
	public static SimpleBrain newFromFile(String fileName)
	{
		SimpleBrain temp = new SimpleBrain();
		temp.importBrain(fileName);
		return temp;
	}

}
