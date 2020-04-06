
using Accord.Neuro;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge.Neuro.Learning;
using System.IO;

namespace xamarin_neural_network
{
	public class DeepLearning: CrossValidationPerceptronLearning
	{
		public static double[][] trainingInput2;
		public static double[][] validationInput2;
		public static double[][] testingInput2;
		public static double[][] trainigOutput2;
		public static double[][] validationOutput2;
		public static double[][] testingOutput2;
		private static DeepBeliefNetwork network;
		private static mylearning teacher2;

		public double[][] TrainingInput2
		{
			get { return trainingInput2; }
			set { trainingInput2 = value; }
		}

		public double[][] ValidationInput2
		{
			get { return validationInput2; }
			set { validationInput2 = value; }
		}

		public double[][] TestingInput2
		{
			get { return testingInput2; }
			set { testingInput2 = value; }
		}

		public double[][] TrainingOutput2
		{
			get { return trainigOutput2; }
			set { trainigOutput2 = value; }
		}

		public double[][] ValidationOutput2
		{
			get { return validationOutput2; }
			set { validationOutput2 = value; }
		}

		public double[][] TestingOutput
		{
			get { return testingOutput2; }
			set { testingOutput2 = value; }
		}
		public DeepLearning()
		{ 
		}

		public DeepLearning(double[][] inputs, double[][] outputs, double[][] validationInputs, double[][] validationOutputs, double[][] testInputs, double[][] testOutputs)
		{
			trainingInput2    = inputs;
			trainigOutput2    = outputs;
			validationInput2  = validationInputs;
			validationOutput2 = validationOutputs;
			testingInput2     = testInputs;
			testingOutput2    = testOutputs;

			Console.WriteLine("Deep learning");

			int batchSize = 10;
			int iter1     = 1;
			int iter2     = 10;
			int iterVA    = 1;
			int iterVa1   = 1;
			int iterVa2   = 100;
			int flag1     = 1;
			int flag2     = 1;
			int flagVa1   = 0;
			int flagVa2   = 1;

			int[] hn = { 3,2,1 }; //8-9 slow and bad  better { 7,6,5,4,3,2,1 };  { 6,5,4,3,2,1 }; { 5,4,3,2,1 }; { 4,3,2,1 }; { 3,2,1 };  { 2,1 };

			myAccordSigmoidStochasticFunction function  = new myAccordSigmoidStochasticFunction(2,network,teacher2);

			network  = new DeepBeliefNetwork(function,  inputs.First().Length, hn);

			new GaussianWeights(network, 0.1).Randomize();

			network.UpdateVisibleWeights();

		
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

			//Supervised teacher
			  teacher2 = new mylearning(network,hn);

			Console.WriteLine("Training Deep Belief Network:");
			trainingDeepBelief2(teacher, teacher2, network, batches, iter1, iter2, inputs, outputs, flag1, flag2);
			Console.WriteLine("Validation Deep Belief Network:");
			writePredictedErrors(network, validationInputs, validationOutputs);
			Console.WriteLine("Training-Validation Deep Belief Network:");
			for (int i = 0; i < iterVA;i++)
				trainingValidationDeepBelief(teacher, teacher2, network, batches, iterVa1, iterVa2, inputs, outputs,validationInputs,validationOutputs,flagVa1,flagVa2);
			Console.WriteLine("Testing");
			writePredictedErrors(network, testInputs, testOutputs);

			Console.WriteLine("end");
		}

		public void setWeights(DeepBeliefNetwork network, double[] wH, double[] wV)
		{
			
			int numMachines = network.Machines.Count;

			int[] numHiddenNuerons = new int[numMachines];
			int[] numVisibleNuerons = new int[numMachines];

			for (int i = 0; i < numMachines; i++)
				numHiddenNuerons[i] = network.Machines[0].Hidden.Neurons.Count();

			for (int i = 0; i < numMachines; i++)
				numVisibleNuerons[i] = network.Machines[0].Visible.Neurons.Count();


		//	Console.WriteLine("HIDDEN WEIGHTS");
			for (int i = 0; i < numMachines; i++)
				for (int j = 0; j < numHiddenNuerons[i]; j++)
					for (int k = 0; k < network.Machines[i].Hidden.Neurons[j].Weights.Count(); k++)
					{
						network.Machines[i].Hidden.Neurons[j].Weights[k] = wH[k];
						
					}

		//	Console.WriteLine("VISIBLE WEIGHTS");
			for (int i = 0; i < numMachines; i++)
				for (int j = 0; j < numVisibleNuerons[i]; j++)
					for (int k = 0; k < network.Machines[i].Visible.Neurons[j].Weights.Count(); k++)
					{
						network.Machines[i].Visible.Neurons[j].Weights[k] = wV[k];
					}
		}


		public void writeWeights(DeepBeliefNetwork network)
		{
			
			int numMachines			 	= network.Machines.Count;

			int[] numHiddenNuerons   	= new int[numMachines];
			int[] numVisibleNuerons  	= new int[numMachines];


			List<double> wH = new List<double>();
			List<double> wV = new List<double>();

			wH.Clear();
			wV.Clear();

			for (int i = 0; i < numMachines;i++)
				numHiddenNuerons[i] 	= network.Machines[0].Hidden.Neurons.Count();
			
			for (int i = 0; i < numMachines; i++)
				numVisibleNuerons[i] 	= network.Machines[0].Visible.Neurons.Count();


			Console.WriteLine("HIDDEN WEIGHTS");
			for (int i = 0; i < numMachines; i++) 
				for (int j = 0; j < numHiddenNuerons[i]; j++)
					for (int k = 0; k < network.Machines[i].Hidden.Neurons[j].Weights.Count(); k++)
					{
						Console.WriteLine(network.Machines[i].Hidden.Neurons[j].Weights[k]);
						wH.Add(network.Machines[i].Hidden.Neurons[j].Weights[k]);
						}

			Console.WriteLine("VISIBLE WEIGHTS");
			for (int i = 0; i < numMachines; i++)
				for (int j = 0; j < numVisibleNuerons[i]; j++)
					for (int k = 0; k < network.Machines[i].Visible.Neurons[j].Weights.Count(); k++)
					{
						Console.WriteLine(network.Machines[i].Visible.Neurons[j].Weights[k]);
						wV.Add(network.Machines[i].Visible.Neurons[j].Weights[k]);
					}

		}

		public List<double>  getHiddenWeights(DeepBeliefNetwork network)
		{
			int numMachines         = network.Machines.Count;

			int[] numHiddenNuerons  = new int[numMachines];

			List<double> wH = new List<double>();
	
			wH.Clear();
		
			for (int i = 0; i < numMachines; i++)
				numHiddenNuerons[i] = network.Machines[0].Hidden.Neurons.Count();

			Console.WriteLine("HIDDEN WEIGHTS");
			for (int i = 0; i < numMachines; i++)
				for (int j = 0; j < numHiddenNuerons[i]; j++)
					for (int k = 0; k < network.Machines[i].Hidden.Neurons[j].Weights.Count(); k++)
					{
						Console.WriteLine(network.Machines[i].Hidden.Neurons[j].Weights[k]);
						wH.Add(network.Machines[i].Hidden.Neurons[j].Weights[k]);
					}

			return wH;
		}



		public List<double> getVisibleWeights(DeepBeliefNetwork network)
		{
			int numMachines = network.Machines.Count;

			int[] numVisibleNuerons = new int[numMachines];


			  List<double> wV = new List<double>();

		     wV.Clear();

		
				for (int i = 0; i < numMachines; i++)
					numVisibleNuerons[i] = network.Machines[0].Visible.Neurons.Count();

				Console.WriteLine("VISIBLE WEIGHTS");
				for (int i = 0; i < numMachines; i++)
					for (int j = 0; j < numVisibleNuerons[i]; j++)
						for (int k = 0; k < network.Machines[i].Visible.Neurons[j].Weights.Count(); k++)
						{
							Console.WriteLine(network.Machines[i].Visible.Neurons[j].Weights[k]);
							wV.Add(network.Machines[i].Visible.Neurons[j].Weights[k]);
						}

			return wV;
		}
		/**************************************KINECT******************************************************/

/*
		public void writePredictedErrors(DeepBeliefNetwork  network, double[][] inputs, double[][] outputs)
		{
			Console.WriteLine("----------------------------------------------------------------------------------------");

			double[][] predictedValues= new double[inputs.Length][];

			for (int i = 0; i < inputs.Length; i++)
			{
				predictedValues[i]  = network.Compute(inputs[i]);
				Console.WriteLine(outputs[i][0]+" ->" + " Predicted value : " + predictedValues[i][0]);
			}

		//Console.WriteLine("Correct " + correct + "/" + inputs.Length + ", " + Math.Round(((double)correct / (double)inputs.Length * 100), 2) + "%");
		    TransformData transobj  = new TransformData();

			double[] transOutput    = transobj.multiplyVectorByConst(outputs, 0, 1000);
			double[] transPredicted = transobj.multiplyVectorByConst(predictedValues,0, 1000);

		// Mean error
	    	Error er        = new Error();
		    double absError = er.MeanAbsoluteError(transOutput, transPredicted);

    		Console.WriteLine("MeanAbsoluteError:  " + absError);
			Console.WriteLine("----------------------------------------------------------------------------------------");
		
		}
*/
		/**************************************HEART******************************************************/

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

			double[] transOutput = transobj.multiplyVectorByConst(outputs, 0, 1);
			double[] transPredicted = transobj.multiplyVectorByConst(predictedValues, 0, 1);

			// Mean error
			Error er = new Error();
			double absError = er.MeanAbsoluteError(transOutput, transPredicted);

			Console.WriteLine("MeanAbsoluteError:  " + absError);
			Console.WriteLine("----------------------------------------------------------------------------------------");

		}



		public void trainingValidationDeepBelief(DeepBeliefNetworkLearning teacher, mylearning teacher2, DeepBeliefNetwork network, double[][][] batches, int iter1, int iter2, double[][] inputs, double[][] outputs, double[][] validationInputs, double[][] validationOutputs, int flag1, int flag2)
		{
			Console.WriteLine("Training validation");
			
			IList<int> appendedUnsortListIndex = new List<int>();

			int tr = inputs.Length;

			double[][] newTrainigInput = new double[tr][];
			double[][] newTrainigOutput = new double[tr][];

			IList<int> appendedList = new List<int>();

			appendedList = KfoldCrossValidation.createListint(inputs.Length + validationInputs.Length);
			appendedUnsortListIndex = KfoldCrossValidation.unsortList(appendedList);

			newTrainigInput = trainingValidationData(inputs, validationInputs, appendedUnsortListIndex);
			newTrainigOutput = trainingValidationData(outputs, validationOutputs, appendedUnsortListIndex);

			trainingInput2 = newTrainigInput;
			trainigOutput2 = newTrainigOutput;

			
			trainingDeepBelief2(teacher, teacher2, network, batches, iter1, iter2, newTrainigInput, newTrainigOutput, flag1, flag2);

		}

		public void trainingDeepBelief2(DeepBeliefNetworkLearning teacher, mylearning teacher2, DeepBeliefNetwork network, double[][][] batches, int iter1, int iter2, double[][] inputs, double[][] outputs, int flag1, int flag2)
		{
			

			// Learning data for the specified layer.
			double[][][] layerData;

			if (flag1 == 1)
			{
				// Unsupervised learning on each hidden layer, except for the output layer.

				for (int layerIndex = 0; layerIndex < network.Machines.Count - 1; layerIndex++)
				{
					teacher.LayerIndex = layerIndex;
					layerData = teacher.GetLayerInput(batches);
					for (int i = 0; i < iter1; i++)
					{

						double error1 = teacher.RunEpoch(layerData) / inputs.Length;
						//Console.WriteLine(i + " Epoch completed with an error1: " + error1);
					}
				}

				Console.WriteLine("Teacher 1:");
				writePredictedErrors(network, inputs, outputs);
			}


			// Run supervised learning.
			if (flag2 == 1)
			{
				for (int i = 0; i < iter2; i++)
				{
					double error2 = teacher2.RunEpoch2(inputs, outputs,i+1) / inputs.Length;
					//Console.WriteLine(i + " Epoch completed with an error2: " + error2);
				}

				Console.WriteLine("Teacher 2:");
				writePredictedErrors(network, inputs, outputs);
				//writePredictedErrors(network, validationInputs, validationOutputs);
			}
			/******************************************************************************************/


		}




	}
}
