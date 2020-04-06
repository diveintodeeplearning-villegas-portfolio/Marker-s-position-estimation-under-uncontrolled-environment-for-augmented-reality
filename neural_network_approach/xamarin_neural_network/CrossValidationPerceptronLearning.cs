using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AForge.Neuro;
using AForge.Neuro.Learning;
using MathNet;

namespace xamarin_neural_network
{
	public class CrossValidationPerceptronLearning
	{
		public CrossValidationPerceptronLearning()
		{
		}

		/***********************TRAINING AND VALIDATION --TR+VA+LEARNING*****************************************************************/

		//The validation set Va is then used to compute the classification error E = M - Y = teacher.runepoch(theinput_va,theutput_va) with (teacher_training) which uses --> (network_training)
		//using the weights resulting from the training_-->(network_training)
		// where M is the expected output vector taken from the validation set ->(theoutput_va)
		// and Y is the computed output resulting from the classification (Y = W * X).-->network_training.compute(theinput_va)
		// If the error is higher than a user-defined threshold then the whole training-validation epoch is repeated. 
		//This training phase ends when the error computed using the validation set is deemed low enough.

		//Smart Training
		//Now, a smart ruse here is to randomly select which samples to use for training and validation
		//from the total set Tr + Va at each epoch iteration. This ensures that the network will not over-fit 

		/*****************************K-FOLD-CROSS-VALIDATION********************************************************/

		public void trainingValidation(PerceptronLearning teacher, double desiredError, int countEpoch, double[][] theinput, double[][] theoutput, double[][] theinput_va, double[][] theoutput_va, Network network, String FileNetwork, String FileResultsVa)
		{
			double error         = 100000;
			double minError      = 100000;
			double[][] netout_tr = new double[theinput.Length][];
			double[][] netout_va = new double[theinput_va.Length][];
			double absError      = 0;
			double[] realZ       = new double[theinput_va.Length];
			double[] predicted   = new double[theinput_va.Length];


			TransformData transobj = new TransformData();

			/*****************************K-FOLD-CROSS-VALIDATION********************************************************/

			IList<int> appendedUnsortListIndex = new List<int>();

			int tr = theinput.Length;

			double[][] newTrainigInput  = new double[tr][];
			double[][] newTrainigOutput = new double[tr][];

			IList<int> appendedList = new List<int>();

			appendedList            = KfoldCrossValidation.createListint(theinput.Length + theinput_va.Length);
			appendedUnsortListIndex = KfoldCrossValidation.unsortList(appendedList);

			newTrainigInput  = trainingValidationData(theinput,  theinput_va,  appendedUnsortListIndex);
			newTrainigOutput = trainingValidationData(theoutput, theoutput_va, appendedUnsortListIndex);

			NNPerceptronLearning.trainigInput2  = newTrainigInput;
			NNPerceptronLearning.trainigOutput2 = newTrainigOutput;
			/*********************************************************************************************************/

			//Validation process                        
			Console.WriteLine("Validation Epoch:");


			for (int j = 0; j < theinput_va.Length; j++) netout_va[j] = network.Compute(theinput_va[j]);


			realZ     = transobj.multiplyVectorByConst(theoutput_va, 0, 1000);
			predicted = transobj.multiplyVectorByConst(netout_va, 0, 1000);

			Error e  = new Error();
			absError = e.MeanAbsoluteError(realZ, predicted);
			Console.WriteLine("----------------------------------------------------------------------------------------");
			Console.WriteLine(countEpoch + " Validation Epoch compleated with error: " + error + "   Validation MeanAbsoluteError:  " + absError);
			Console.WriteLine("----------------------------------------------------------------------------------------");

			//TR+VA learning process                        
			Console.WriteLine("Press ESC to stop");
			do
			{
				//	while (!Console.KeyAvailable)//&&(minError==error))
				//	{
				error = teacher.RunEpoch(newTrainigInput, newTrainigOutput);
				if (error < minError) minError = error;

				Console.WriteLine(countEpoch + " Validation + Training Epoch compleated with error: " + error);

				appendedUnsortListIndex = KfoldCrossValidation.unsortList(appendedList);

				newTrainigInput  = trainingValidationData(theinput, theinput_va, appendedUnsortListIndex);
				newTrainigOutput = trainingValidationData(theoutput, theoutput_va, appendedUnsortListIndex);

				NNPerceptronLearning.trainigInput2 = newTrainigInput;
				NNPerceptronLearning.trainigOutput2 = newTrainigOutput;

				countEpoch++;

				//}
			} while (error > desiredError);//((error < desiredError) && (Console.ReadKey(true).Key != ConsoleKey.Escape));

			for (int j = 0; j < newTrainigInput.Length; j++) netout_tr[j] = network.Compute(newTrainigInput[j]);//new double[4] { 43.84087777, 23.0565944, 367.1607626, 238  }


			realZ     = transobj.multiplyVectorByConst(newTrainigOutput, 0, 1000);
			predicted = transobj.multiplyVectorByConst(netout_tr, 0, 1000);

			absError = e.MeanAbsoluteError(realZ, predicted);
			Console.WriteLine("----------------------------------------------------------------------------------------");
			Console.WriteLine(countEpoch + " TR+VA Epoch compleated with error: " + error + "   TR+VA MeanAbsoluteError:  " + absError);

			Perceptron net = new Perceptron();
			//write training of the network of training
			net.writeTrainingAndTesting(network, newTrainigInput, FileResultsVa);
			Console.WriteLine("----------------------------------------------------------------------------------------");
		}



		/****************************TESTING***************************************************************************************/

		public void testing(double[][] theinput, double[][] theoutput, Network network, String FileResultsTe)
		{
			double error         = 100000;
			double[][] netout    = new double[theinput.Length][];
			double absError      = 0;
			double[] realZ       = new double[theinput.Length];
			double[] predicted   = new double[theinput.Length];

			TransformData transobj = new TransformData();

			for (int j = 0; j < theinput.Length; j++) netout[j] = network.Compute(theinput[j]);

			realZ      = transobj.multiplyVectorByConst(theoutput, 0, 1000);
			predicted  = transobj.multiplyVectorByConst(netout, 0, 1000);

			Error e  = new Error();
			absError = e.MeanAbsoluteError(realZ, predicted);
			Console.WriteLine("----------------------------------------------------------------------------------------");
			Console.WriteLine(" Testing compleated with error: " + error + "   Testing MeanAbsoluteError:  " + absError);

			Perceptron net = new Perceptron();
			//write training of the network of training
			net.writeTrainingAndTesting(network, theinput, FileResultsTe);
			Console.WriteLine("----------------------------------------------------------------------------------------");

		}


		/***************************TRAINING LEARNING***************************************************************************************/
		public void training(PerceptronLearning teacher, double desiredError, int countEpoch, double[][] theinput, double[][] theoutput, Network network, String FileNetwork)
		{
			double error       = 100000;
			double minError    = 100000;
			double[][] netout  = new double[theinput.Length][];
			double absError    = 0;
			double[] realZ     = new double[theinput.Length];
			double[] predicted = new double[theinput.Length];


			TransformData transobj = new TransformData();

			Console.WriteLine("----------------------------------------------------------------------------------------");
			//learning process                        
			Console.WriteLine("Press ESC to stop");
			do
			{
				//while (!Console.KeyAvailable)//&&(minError==error))
				//	{

				error = teacher.RunEpoch(theinput, theoutput);
				if (error < minError) minError = error;

				Console.WriteLine(countEpoch + " Epoch compleated with error: " + error);


				countEpoch++;

				//}
			} while (error > desiredError);//((error < desiredError) && (Console.ReadKey(true).Key != ConsoleKey.Escape));

			for (int j = 0; j < theinput.Length; j++) netout[j] = network.Compute(theinput[j]);//new double[4] { 43.84087777, 23.0565944, 367.1607626, 238  }


			realZ     = transobj.multiplyVectorByConst(theoutput, 0, 1000);
			predicted = transobj.multiplyVectorByConst(netout, 0, 1000);

			Error e  = new Error();
			absError = e.MeanAbsoluteError(realZ, predicted);

			Console.WriteLine(countEpoch + " Epoch compleated with error: " + error + "   MeanAbsoluteError:  " + absError);
			Console.WriteLine("----------------------------------------------------------------------------------------");
		}
		/*********************************************************************************************************/

		public double[][] trainingValidationData(double[][] thedata, double[][] thedata_va, IList<int> appendedUnsortListIndex)
		{
			/*****************************K-FOLD-CROSS-VALIDATION********************************************************/

			int tr = thedata.Length;
			int va = thedata_va.Length;

			double[][] newTrainigInput = new double[tr][];

			double[][] appendedTrVaInput = new double[tr + va][];

			appendedTrVaInput = KfoldCrossValidation.appendMatrixs(thedata, thedata_va);

			appendedTrVaInput = KfoldCrossValidation.unsortedMatrix(appendedTrVaInput, appendedUnsortListIndex);

			newTrainigInput = KfoldCrossValidation.deleteMatrixLasts(appendedTrVaInput, va);

			return newTrainigInput;


		}
		/***********************TRAININGandTESTING*****************************************************************/

		public void writeTrainingAndTesting(Network network, double[][] input, String fileResults)
		{
			ReaderExcel writer   = new ReaderExcel();

			double[][] netout    = new double[input.Length][];

			String[][] StrNetout = new string[input.Length][];

			Console.WriteLine("Training phase->Transformed Input and Output data:");
			int j = 0;
			foreach (double[] i in input)
			{
				netout[j] = network.Compute(i);//new double[4] { 43.84087777, 23.0565944, 367.1607626, 238  }
				Console.WriteLine(i[0] + " " + i[1] + " " + i[2] + " " + i[3] + "->" + netout[j][0]);
				j++;
			}

			//Writing in excel the results
			TransformData transobj = new TransformData();
			StrNetout = transobj.doubleArrayToString(netout);
			writer.WriterExcel(fileResults, StrNetout);
		}
		/***********************wWEIGHTS*****************************************************************/
		/*
		public void writeWeights(Network network)
		{

			double weight1 = network.Layers[0].Neurons[0].Weights[0];
			double weight2 = network.Layers[0].Neurons[0].Weights[1];
			double weight3 = network.Layers[0].Neurons[0].Weights[2];
			double weight4 = network.Layers[0].Neurons[0].Weights[3];

			Console.WriteLine("Weight 1: " + weight1 + " Weight 2: " + weight2 + " Weight 3: " + weight3 + " Weight 4: " + weight4);
		}
		*/
		/***********************BACK LEARNING*****************************************************************/

		/*
		public void backlearning(BackPropagationLearning teacher, double desiredError, int countEpoch, double[][] theinput, double[][] theoutput, Network network, String FileNetwork)
		{
			double error = 100000;

			//learning process                        
			do
			{
				error = teacher.RunEpoch(theinput, theoutput);
				Console.WriteLine(countEpoch + " Epoch compleated with error: " + error);
				Perceptron net = new Perceptron();
				net.writeWeights(network);

				countEpoch++;

			} while (error >= desiredError);

			//save data of the network
			network.Save(FileNetwork);

		}
		*/
	}
}
