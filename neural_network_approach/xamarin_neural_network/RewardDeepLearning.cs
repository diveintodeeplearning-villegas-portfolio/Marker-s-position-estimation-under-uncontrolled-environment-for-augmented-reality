
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;


namespace xamarin_neural_network
{
	public class RewardDeepLearning 
	{
		public static double[][]         trainingInput2;
		public static double[][]         trainigOutput2;
		public static double             reward;
		public static State              minState  = new State(); 
		public static State              lastState = new State(); 
		private static DeepBeliefNetwork network;
		private static mylearning        teacher2;


		public double[][] TrainingInput2
		{
			get { return trainingInput2; }
			set { trainingInput2 = value; }
		}

		public double[][] TrainingOutput2
		{
			get { return trainigOutput2; }
			set { trainigOutput2 = value; }
		}

		public double Reward
		{
			get { return reward; }
			set { reward = value; }
		}

		public State MinState
		{
			get { return minState; }
			set { minState= value; }
		}

		public State LastState
		{
			get { return lastState; }
			set { lastState = value; }
		}


		public RewardDeepLearning(double[][] inputs, double[][] outputs, Action a1, int[] hn, double increment)
		{
			//Console.WriteLine("increment :" + increment);

			trainingInput2 = inputs;
			trainigOutput2 = outputs;


			//Console.WriteLine("Deep learning");

			int batchSize = 10;
			int iter1 = 10 * 5;
			int iter2 = 10 * 5;
		

			myAccordSigmoidStochasticFunction function = new myAccordSigmoidStochasticFunction(2, network, teacher2);

			//int[] hn = { 5, 2, 1 };
			network = new DeepBeliefNetwork(function, inputs.First().Length, hn); //1

			//network = new DeepBeliefNetwork(function, inputs.First().Length, 1);

			//new GaussianWeights(network, 0.1).Randomize();

			setWeightsTresholdIncrement(network, a1.hiddenWeights, a1.visibleWeights,a1.hiddenTreshold,a1.visibleTreshold, increment);
			//network.UpdateVisibleWeights();

			DeepBeliefNetworkLearning teacher = new DeepBeliefNetworkLearning(network)
			{
				Algorithm = (h, v, i) => new ContrastiveDivergenceLearning(h, v)
				{
					LearningRate = 0.1,
					Momentum = 0.5,
					Decay = 0.001,
				}
			};


			// Setup batches of input for learning.
			int batchCount = Math.Max(1, inputs.Length / batchSize);
			// Create mini-batches to speed learning.
			int[] groups = Accord.Statistics.Classes.Random(inputs.Length, batchCount);
			double[][][] batches = inputs.Subgroups(groups);
			// Learning data for the specified layer.
			teacher2 = new mylearning(network,hn);

			//Console.WriteLine("Training Deep Belief Network:");
			double error = trainingDeepBelief2(teacher, teacher2,  batches, iter1, iter2, inputs, outputs);
			reward = error;

			string[] arrayActions = teacher2.TakenActions.ToArray();

			//foreach (string act in arrayActions) Console.WriteLine(act);
		}

		public void setWeightsTresholdIncrement(DeepBeliefNetwork network, List<double> wH, List<double> wV, List<double> tH, List<double> tV, double increment)
		{

			int numMachines = network.Machines.Count;

			int[] numHiddenNuerons  = new int[numMachines];
			int[] numVisibleNuerons = new int[numMachines];


			for (int i = 0; i < numMachines; i++)
				numHiddenNuerons[i] = network.Machines[0].Hidden.Neurons.Count();

			for (int i = 0; i < numMachines; i++)
				numVisibleNuerons[i] = network.Machines[0].Visible.Neurons.Count();

			int listWeights= 0;
			int listTreshold = 0;

			//Console.WriteLine("HIDDEN WEIGHTS");
			for (int i = 0; i < numMachines; i++)
			{
				for (int j = 0; j < network.Machines[i].Hidden.Neurons.Length; j++)
				{
					for (int k = 0; k < network.Machines[i].Hidden.Neurons[j].Weights.Length; k++)
					{
						network.Machines[i].Hidden.Neurons[j].Weights[k] = wH[listWeights] + increment;
						listWeights++;
					}

					network.Machines[i].Hidden.Neurons[j].Threshold = tH[listTreshold];
					listTreshold++;
				}
			}

			listWeights = 0;
			listTreshold = 0;
			//Console.WriteLine("VISIBLE WEIGHTS");
			for (int i = 0; i < numMachines; i++)
			{
				for (int j = 0; j < network.Machines[i].Visible.Neurons.Length; j++)
				{
					for (int k = 0; k < network.Machines[i].Visible.Neurons[j].Weights.Count(); k++)
					{
						network.Machines[i].Visible.Neurons[j].Weights[k] = wV[listWeights] + increment;
						listWeights++;
					}

					network.Machines[i].Visible.Neurons[j].Threshold = tV[listTreshold];
					listTreshold++;
				}
			}
		}




		public void writeWeights(DeepBeliefNetwork network)
		{

			int numMachines = network.Machines.Count;

			int[] numHiddenNuerons  = new int[numMachines];
			int[] numVisibleNuerons = new int[numMachines];


			List<double> wH = new List<double>();
			List<double> wV = new List<double>();

			wH.Clear();
			wV.Clear();

			for (int i = 0; i < numMachines; i++)
				numHiddenNuerons[i] = network.Machines[0].Hidden.Neurons.Count();

			for (int i = 0; i < numMachines; i++)
				numVisibleNuerons[i] = network.Machines[0].Visible.Neurons.Count();


			//Console.WriteLine("HIDDEN WEIGHTS");
			for (int i = 0; i < numMachines; i++)
				for (int j = 0; j < network.Machines[i].Hidden.Neurons.Length; j++)
					for (int k = 0; k < network.Machines[i].Hidden.Neurons[j].Weights.Count(); k++)
					{
						//	Console.WriteLine(i +" "+ j + " "+k +" "+ network.Machines[i].Hidden.Neurons[j].Weights[k]);
						wH.Add(network.Machines[i].Hidden.Neurons[j].Weights[k]);
					}

			//Console.WriteLine("VISIBLE WEIGHTS");
			for (int i = 0; i < numMachines; i++)
				for (int j = 0; j < network.Machines[i].Visible.Neurons.Length; j++)
					for (int k = 0; k < network.Machines[i].Visible.Neurons[j].Weights.Length; k++)
					{
						//Console.WriteLine(i + " " + j + " " + k + " " + network.Machines[i].Visible.Neurons[j].Weights[k]);
						wV.Add(network.Machines[i].Visible.Neurons[j].Weights[k]);
					}
		}

		public List<double> getHiddenWeights(DeepBeliefNetwork network)
		{

			int numMachines = network.Machines.Count;

			int[] numHiddenNuerons = new int[numMachines];

			//double[][][] hiddenWeights = new double[numMachines][][];

			List<double> wH = new List<double>();

			wH.Clear();

			for (int i = 0; i < numMachines; i++)
				numHiddenNuerons[i] = network.Machines[0].Hidden.Neurons.Count();


			for (int i = 0; i < numMachines; i++)
			{
				for (int j = 0; j < network.Machines[i].Hidden.Neurons.Length; j++)
				{
					foreach (double w in network.Machines[i].Hidden.Neurons[j].Weights)
						wH.Add(w);
				}
			}


			return wH;
		}



		public List<double> getVisibleWeights(DeepBeliefNetwork network)
		{
			int numMachines = network.Machines.Count;

			int[] numVisibleNuerons = new int[numMachines];

			//double[][][] visibleWeights = new double[numMachines][][];

			List<double> wV = new List<double>();

			wV.Clear();

			for (int i = 0; i < numMachines; i++)
				numVisibleNuerons[i] = network.Machines[0].Visible.Neurons.Count();


			for (int i = 0; i < numMachines; i++)
				for (int j = 0; j < network.Machines[i].Visible.Neurons.Length; j++)
					foreach (double w in network.Machines[i].Visible.Neurons[j].Weights)
						wV.Add(w);



			return wV;
		}

		public List<double> getHiddenTreshold(DeepBeliefNetwork network)
		{

			int numMachines = network.Machines.Count;

			int[] numHiddenNuerons = new int[numMachines];

			//double[][][] hiddenWeights = new double[numMachines][][];

			List<double> hT = new List<double>();

			hT.Clear();

			for (int i = 0; i < numMachines; i++)
				numHiddenNuerons[i] = network.Machines[0].Hidden.Neurons.Count();


			//Console.WriteLine("HIDDEN WEIGHTS");
			for (int i = 0; i < numMachines; i++)
			{
				for (int j = 0; j < network.Machines[i].Hidden.Neurons.Length; j++)
					//for (int k = 0; k < network.Machines[i].Hidden.Neurons[j].Weights.Length; k++)
					//{
					//Console.WriteLine(i + " " + j + " " + k + " " + network.Machines[i].Hidden.Neurons[j].Weights[k]);
					hT.Add(network.Machines[i].Hidden.Neurons[j].Threshold);
				//hiddenWeights[i][j][k] = network.Machines[i].Hidden.Neurons[j].Weights[k];
				//}
			}

			return hT;
		}



		public List<double> getVisibleTreshold(DeepBeliefNetwork network)
		{
			int numMachines = network.Machines.Count;

			int[] numVisibleNuerons = new int[numMachines];

			//double[][][] visibleWeights = new double[numMachines][][];

			List<double> vT = new List<double>();

			vT.Clear();

			for (int i = 0; i < numMachines; i++)
				numVisibleNuerons[i] = network.Machines[0].Visible.Neurons.Count();

			//Console.WriteLine("VISIBLE WEIGHTS");
			for (int i = 0; i < numMachines; i++)
			{
				for (int j = 0; j < network.Machines[i].Visible.Neurons.Length; j++)
					//for (int k = 0; k < network.Machines[i].Visible.Neurons[j].Weights.Count(); k++)
					//{
					//Console.WriteLine(i + " " + j + " " + k + " " + network.Machines[i].Visible.Neurons[j].Weights[k]);
					vT.Add(network.Machines[i].Visible.Neurons[j].Threshold);
				//visibleWeights[i][j][k] = network.Machines[i].Visible.Neurons[j].Weights[k];
				//}
			}

			return vT;
		}




	

	




		public void writePredictedErrors(DeepBeliefNetwork network, double[][] inputs, double[][] outputs)
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

			double[] transOutput    = transobj.multiplyVectorByConst(outputs, 0, 1000);
			double[] transPredicted = transobj.multiplyVectorByConst(predictedValues, 0, 1000);

			// Mean error
			Error er        = new Error();
			double absError = er.MeanAbsoluteError(transOutput, transPredicted);

			Console.WriteLine("MeanAbsoluteError:  " + absError);
			Console.WriteLine("----------------------------------------------------------------------------------------");

		}



		public double trainingDeepBelief2(DeepBeliefNetworkLearning teacher, mylearning teacher2,  double[][][] batches, int iter1, int iter2, double[][] inputs, double[][] outputs)
		{
			    double minError = 10000000000000000;
				double error2 = 0;
				// Run supervised learning.

				for (int i = 0; i < iter2; i++)
				{
					error2 = teacher2.RunEpoch3(inputs, outputs, i + 1) / inputs.Length;
			
				if (error2 < minError)
				{
					minState.HiddenWeights   = getHiddenWeights(network);
					minState.VisibleWeights  = getVisibleWeights(network);
					minState.HiddenTreshold  = getHiddenTreshold(network);
					minState.VisibleTreshold = getVisibleTreshold(network);
					minState.Reward          = error2;
					minState.Epochs          = i;
				}
					
					
					lastState.HiddenWeights   = getHiddenWeights(network);
					lastState.VisibleWeights  = getVisibleWeights(network);
				    lastState.HiddenTreshold  = getHiddenTreshold(network);
					lastState.VisibleTreshold = getVisibleTreshold(network);
				    lastState.Reward          = error2;
				    lastState.Epochs          = i;

				}

				//Console.WriteLine("Reward Teacher 2:");
				//writePredictedErrors(network, inputs, outputs);
				//writePredictedErrors(network, validationInputs, validationOutputs);

				/******************************************************************************************/

			return minState.Reward;

		}


	}
}
