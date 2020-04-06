using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AForge.Neuro;
using AForge.Neuro.Learning;
using MathNet;

namespace xamarin_neural_network
{
	public class NNPerceptronLearning
	{
		public static double[][] trainigInput2;
		public static double[][] validationInput2;
		public static double[][] testingInput2;
		public static double[][] trainigOutput2;
		public static double[][] validationOutput2;
		public static double[][] testingOutput2;

		public double[][] TrainingInput2
		{
			get { return trainigInput2; }
			set { trainigInput2 = value; }
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

		public NNPerceptronLearning(int file, LightRange R3, double[][] trainigInput, double[][] trainigOutput)
		{
			trainigInput2 = trainigInput;
			trainigOutput2 = trainigOutput;


			Console.WriteLine("Perceptron Learning");

			//                          0               1                2            3           4         
			String[] FileName = { "LOGITECH1data", "LOGITECH2data", "LOGITECH3data", "CAM2data", "KINECTdata" };
			String[] strRange = { "ALL", "NEWR1", "NEWR2", "NEWR3TRAINING", "NEWRTRAINING4", "NEWR5", "NEWR6", "NEWR7", "ALL" };
			String FileNetwork = "Network/" + strRange[R3.getLightIndex()] + FileName[file] + "_networkTr-6.bin";
			String FileResults = "Results/Results" + strRange[R3.getLightIndex()] + FileName[file] + "Tr-6.csv";


			//initialize the iterations
			int countEpoch   = 0;
			double error     = 0.013;
			int initialInput = 4;
			int hiddenLayers = 1;
			//double alphaSigmoidFunction = 2;
			double learningRate = 0.1;


			//network Perceptron
			ActivationNetwork network = new ActivationNetwork(
					new myUpdateBipolarSigmodFunction(2),//myBipolarSigmodFunction(2,0.250),//myActivationFunction(), //new SigmoidFunction(alphaSigmoidFunction),
					initialInput,
					hiddenLayers
				);
			//teacher and learning
			PerceptronLearning teacher = new PerceptronLearning(network);

			teacher.LearningRate = learningRate;


			Perceptron net = new Perceptron();

			//learning process    
			net.training(teacher, error, countEpoch, trainigInput, trainigOutput, network, FileNetwork);//net.learning( teacher, error,  countEpoch ,theinput, theoutput, net, network,FileNetwork);

			//save data of the network of training
			network.Save(FileNetwork);

			//write training of the network of training
			net.writeTrainingAndTesting(network, trainigInput, FileResults);//net.writeTrainingAndTesting(network, theinput, FileResults);

		}


		public NNPerceptronLearning(int file,LightRange R3,double[][] trainigInput, double[][]trainigOutput,double[][] validationInput, double[][] validationOutput, double[][] testingInput, double[][] testingOutput)
		{
			trainigInput2     = trainigInput;
			trainigOutput2    = trainigOutput;
			validationInput2  = validationInput;
			validationOutput2 = validationOutput;
			testingInput2     = testingInput;
			testingOutput2    = testingOutput;


			Console.WriteLine("Perceptron Learning");

			//                          0               1                2            3           4         
			String[] FileName    = { "LOGITECH1data", "LOGITECH2data", "LOGITECH3data", "CAM2data", "KINECTdata" };
			String[] strRange    = { "ALL", "NEWR1", "NEWR2", "NEWR3TRAINING", "NEWRTRAINING4", "NEWR5", "NEWR6", "NEWR7", "ALL" };
			String FileNetwork   = "Network/" + strRange[R3.getLightIndex()]        + FileName[file] + "_networkTr-6.bin";
			String FileNetworkVa = "Network/" + strRange[R3.getLightIndex()]        + FileName[file] + "_networkVA-6.bin";
			String FileResults   = "Results/Results" + strRange[R3.getLightIndex()] + FileName[file] + "Tr-6.csv";
			String FileResultsVa = "Results/Results" + strRange[R3.getLightIndex()] + FileName[file] + "Va-6.csv";
			String FileResultsTe = "Results/Results" + strRange[R3.getLightIndex()] + FileName[file] + "Te.csv";
	

			//initialize the iterations
			int    countEpoch        = 0;
			double error          = 0.013;
			double errorVa        = 0.010;
			int    initialInput      = 4;
			int    hiddenLayers      = 1;
			//double alphaSigmoidFunction = 2;
			double learningRate   = 0.1;
			double learningRateVa = 0.001;


			//network Perceptron
			ActivationNetwork network = new ActivationNetwork(
					new myUpdateBipolarSigmodFunction(2),//myBipolarSigmodFunction(2,0.250),//myActivationFunction(), //new SigmoidFunction(alphaSigmoidFunction),
					initialInput,
					hiddenLayers
				);
			//teacher and learning
			PerceptronLearning teacher = new PerceptronLearning(network);

			teacher.LearningRate = learningRate;


			Perceptron net = new Perceptron();

			//learning process    
			net.training(teacher, error, countEpoch, trainigInput, trainigOutput, network, FileNetwork);//net.learning( teacher, error,  countEpoch ,theinput, theoutput, net, network,FileNetwork);

			//save data of the network of training
			network.Save(FileNetwork);

			//write training of the network of training
			net.writeTrainingAndTesting(network, trainigInput, FileResults);//net.writeTrainingAndTesting(network, theinput, FileResults);

			// start TR + VA
			teacher.LearningRate = learningRateVa;
			net.trainingValidation(teacher, errorVa, countEpoch, trainigInput, trainigOutput, validationInput, validationOutput, network, FileNetworkVa, FileResultsVa);

			//save data of the network of validation
			network.Save(FileNetworkVa);

			//Testing the data 
			net.testing(testingInput, testingOutput, network, FileResultsTe);

		}
	}
}
