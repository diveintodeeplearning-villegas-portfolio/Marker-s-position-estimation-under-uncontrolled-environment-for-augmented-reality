using System;
using AForge;
using Accord.Neuro.Neurons;
using Accord.Neuro.Networks;
using Accord.Neuro.Learning;
using Accord.Statistics.Distributions.Univariate;
using Accord.Math.Random;
using Accord.Math;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge.Neuro.Learning;
using System.IO;




namespace xamarin_neural_network
{
	[Serializable]
	public class ActionsQLearning : DeepLearning
	{
		public DeepBeliefNetwork net;
		public Accord.Neuro.Learning.ParallelResilientBackpropagationLearning teacher;

		public ActionsQLearning()
		{
			
		}

		public double action(double rule, DeepBeliefNetwork network, Accord.Neuro.Learning.ParallelResilientBackpropagationLearning teacher2)
		{
			double qvalue              = 0;
			this.net                   = network;
			this.teacher               = teacher2;

			double[][] inputs  = DeepLearning.trainingInput2;
			double[][] outputs = DeepLearning.trainigOutput2;

			double[][] predictedValues = new double[inputs.Length][];


			int   numMachines       = net.Machines.Count;
			int[] numHiddenNuerons  = new int[numMachines];
			int[] numVisibleNuerons = new int[numMachines];

			double[] initialHiddenWeights  = getHiddenWeights(net).ToArray();
			double[] initialVisibleWeights = getVisibleWeights(net).ToArray();

			double[] a1hiddenWeights  = initialHiddenWeights;
			double[] a1VisibleWeights = initialVisibleWeights;

			for (int i = 0; i < a1hiddenWeights.Length; i++)   a1hiddenWeights[i]  += rule;
			for (int i = 0; i < a1VisibleWeights.Length; i++) a1VisibleWeights[i]  += rule;

			setWeights(net, a1hiddenWeights, a1VisibleWeights);

			for (int i = 0; i < inputs.Length; i++)
			{
				predictedValues[i] = net.Compute(inputs[i]);
				//Console.WriteLine(outputs[i][0] + " ->" + " Predicted value : " + predictedValues[i][0]);
			}


			//Console.WriteLine("Correct " + correct + "/" + inputs.Length + ", " + Math.Round(((double)correct / (double)inputs.Length * 100), 2) + "%");
			TransformData transobj  = new TransformData();

			double[] transOutput    = transobj.multiplyVectorByConst(outputs, 0, 1000);
			double[] transPredicted = transobj.multiplyVectorByConst(predictedValues, 0, 1000);

			// Mean error
			Error  er       = new Error();
			double absError = er.MeanAbsoluteError(transOutput, transPredicted);

			qvalue = 1 / absError;

			//Console.WriteLine("MeanAbsoluteError:  " + absError);
			//Console.WriteLine("----------------------------------------------------------------------------------------");

			setWeights(net, initialHiddenWeights, initialVisibleWeights);

			return qvalue;

			/*
			 *         public void Randomize()
        {
            foreach (ActivationLayer layer in network.Layers)
            {
                foreach (ActivationNeuron neuron in layer.Neurons)
                {
                    for (int i = 0; i < neuron.Weights.Length; i++)
                        neuron.Weights[i] = random.Next();
                    if (UpdateThresholds)
                        neuron.Threshold = random.Next();
                }
            }
        }
			*/
		}
	}
}
