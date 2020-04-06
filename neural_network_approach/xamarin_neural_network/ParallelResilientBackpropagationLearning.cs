using System;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using Accord.Math;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge.Neuro.Learning;
using System.IO;
using Accord.Neuro;

namespace xamarin_neural_network
{
	public class ParallelResilientBackpropagationLearning
	{
		public ParallelResilientBackpropagationLearning(double[][] inputs, double[][] outputs, double[][] testInputs, double[][] testOutputs)
		{
			Console.WriteLine("Deep learning");

			// create neural network
			Accord.Neuro.ActivationNetwork network = new Accord.Neuro.ActivationNetwork(
				new SigmoidFunction(2), //new myUpdateBipolarSigmodFunction(2),
				4, // two inputs in the network
				2, // two neurons in the first layer
				1); // one neuron in the second layer

			// create teacher
			var teacher = new Accord.Neuro.Learning.ParallelResilientBackpropagationLearning(network);

			// loop
			double error = 1;

			while (error > 0.00001)
			{
				// run epoch of learning procedure
				error = teacher.RunEpoch(inputs, outputs);
				Console.WriteLine("Error in epoch: " + error);
				// check error value to see if we need to stop
				// ...
			}

			writePredictedErrors(network, inputs, outputs);
			writePredictedErrors(network, testInputs, testOutputs);


		}

		public void writePredictedErrors(Accord.Neuro.ActivationNetwork network, double[][] inputs, double[][] outputs)
		{
			Console.WriteLine("----------------------------------------------------------------------------------------");

			double[][] predictedValues = new double[inputs.Length][];

			for (int i = 0; i < inputs.Length; i++)
			{
				predictedValues[i] = network.Compute(inputs[i]);
				Console.WriteLine(outputs[i][0] + " ->" + " Predicted value : " + predictedValues[i][0]);
			}

			//Console.WriteLine("Correct " + correct + "/" + inputs.Length + ", " + Math.Round(((double)correct / (double)inputs.Length * 100), 2) + "%");
			TransformData transobj = new TransformData();

			double[] transOutput = transobj.multiplyVectorByConst(outputs, 0, 1000);
			double[] transPredicted = transobj.multiplyVectorByConst(predictedValues, 0, 1000);

			// Mean error
			Error er = new Error();
			double absError = er.MeanAbsoluteError(transOutput, transPredicted);

			Console.WriteLine("MeanAbsoluteError:  " + absError);
			Console.WriteLine("----------------------------------------------------------------------------------------");

		}
	}
}

