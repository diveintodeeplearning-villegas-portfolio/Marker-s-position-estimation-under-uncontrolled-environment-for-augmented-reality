using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AForge.Neuro;
using AForge.Neuro.Learning;
using MathNet;


namespace xamarin_neural_network
{
	
	class Perceptron: CrossValidationPerceptronLearning
	{
		public static void Main(string[] args)
		{
			
			int file         = 5;

			// LightRange R3 = new LightRange(1,0,100);
			// LightRange R3 = new LightRange(2,100, 200);
			// LightRange R3 = new LightRange(3,200, 400);
			// LightRange R3 = new LightRange(0, 0, 400);
			   LightRange R3 = new LightRange(0, 0, 1000000000);

			double[] rules  = {1, 1.2, 1.3};


			//                                    0               1                2            3           4        5          6 
			String[] FileName          = { "LOGITECH1data","LOGITECH2data","LOGITECH3data","CAM2data","KINECTdata","Hearta1","Heart1" };
			String[] strRange          = { "ALL", "R1", "R2", "R3","ALL","ALL","ALL","ALL", "ALL" };
			String   FileData          =  "Data/"                                               + FileName[file]  + ".csv";

			String Delimiters          =  "NoComma"; //Comma FOR KINECT //NoComma FOR HEART

			String FileNetwork         =  "Network/"             + strRange[R3.getLightIndex()] + FileName[file]  + "_networkTr.bin";
			String FileNetworkVa       =  "Network/"             + strRange[R3.getLightIndex()] + FileName[file]  + "_networkVA.bin";
			String FileResults         =  "Results/Results"      + strRange[R3.getLightIndex()] + FileName[file]  + "Tr.csv";
			String FileResultsTeLoaded =  "Results/Results" + strRange[R3.getLightIndex()] + FileName[file]  + "TeLoaded.csv"; 

			String saveOrLoad          =  "save";//load
			String FileNetworkLoad     =  FileNetworkVa;
			String Approach            =  "NDL";//DL



			/***********************READING DATA*****************************************************************/

			//Read data from excel "CSV" 'cause it's faster than interop
			ReaderExcel reader = new ReaderExcel(@FileData,Delimiters);

			//GET INDEXES OF LIGHT RANGE
			int [] indexes = reader.getIndexRange(R3);
			foreach(int i in indexes)Console.WriteLine(i);
			Console.ReadLine();


			/*
			 * --------------------------------KINECT---------------------------------------------
			//Read data from excel "CSV" 'cause it's faster than interop
			//double[][] input    =  reader.getRangeIntput(R3);
			  double[][] theinput  =  reader.getRangeIntput(R3);

			  double[][] theoutput = reader.getRangeOutput(R3);

			//Transform data to natural logarithmic and decimal 'cause it is the easiest way to work the data
			TransformData TransData= new TransformData();

			theinput  = TransData.transMatrixToDecimalInput(theinput); //TransData.transMatrixToLn(theinput);
			theoutput = TransData.transMatrixToDecimalOutput(theoutput); //TransData.transMatrixToDecimal(theoutput,0);
			*/
			//---------------------------HEART----------------------------------------------------
			double[][] input     = reader.getInput(35);
			double[][] theinput  = reader.getInput(35);
			double[][] theoutput = reader.getOutput(35);



			/*****************************K-FOLD-CROSS-VALIDATION********************************************************/
		    IList<int>   unsortListIndex = new List<int>();
			//int    df = 1; //Kinect
			int      df = 0; //Heart

			int    tr = Convert.ToInt16(Math.Ceiling(theinput.Length  * 0.50)-df);
			int    va = Convert.ToInt16(Math.Ceiling((theinput.Length * 0.25)));
			int    te = (theinput.Length - tr - va);


			double[][] trainigInput     = new double[tr][];
			double[][] validationInput  = new double[va][];
			double[][] testingInput     = new double[te][];

			double[][] trainigOutput    = new double[tr][];
			double[][] validationOutput = new double[va][];
			double[][] testingOutput    = new double[te][];

			double[][] unsortInput      = new double[theinput.Length][];
			double[][] unsortOutput     = new double[theoutput.Length][];

			unsortListIndex  = KfoldCrossValidation.createList(theinput.Length);
			unsortListIndex  = KfoldCrossValidation.unsortList(unsortListIndex);
			unsortInput      = KfoldCrossValidation.unsortedMatrix(theinput, unsortListIndex);
			unsortOutput     = KfoldCrossValidation.unsortedMatrix(theoutput, unsortListIndex);

			for (int i = 0; i < tr; i++)
			{
				trainigInput[i]   = unsortInput[i];
				trainigOutput[i]  = unsortOutput[i];
			}

			for (int i = 0; i < va; i++)
			{
				validationInput[i]  = unsortInput[tr+i];
				validationOutput[i] = unsortOutput[tr+i];
			}

			for (int i = 0; i < te; i++)
			{
				testingInput[i]  = unsortInput[te + i];
				testingOutput[i] = unsortOutput[te + i];
			}


			/**************************MACHINE LEARNING**********************************************************************/
			if (Approach == "ML")
			{
				double[][] input_mlte = new double[validationInput.Length + testingInput.Length][];
				double[][] output_mlte = new double[validationOutput.Length + testingOutput.Length][];

				input_mlte = KfoldCrossValidation.appendMatrixs(validationInput, testingInput);
				output_mlte = KfoldCrossValidation.appendMatrixs(validationOutput, testingOutput);

				MachineLearning ml = new MachineLearning(trainigInput, trainigOutput, input_mlte, output_mlte);
			}

			/**************************DEEP LEARNING**********************************************************************/
			else if (Approach == "DL")
			{
				DeepLearning dl = new DeepLearning(trainigInput,trainigOutput,validationInput,validationOutput,testingInput,testingOutput);
			}

			else if (Approach == "NDL")
			{
				NormalDeepLearning ndl = new NormalDeepLearning(trainigInput, trainigOutput, validationInput, validationOutput, testingInput, testingOutput);
			}
			else {
				/***********************LOAD PERCEPTRON NETWORK****************************************************************/

				if ( Approach=="NNP" && saveOrLoad != "save")
				{
					Network LoadedNetwork = Network.Load(FileNetworkLoad);
					Perceptron LoadedNet  = new Perceptron();

					LoadedNet.writeTrainingAndTesting(LoadedNetwork, theinput, FileResults);

					Perceptron net = new Perceptron();
					//Testing the data 
					net.testing(theinput, theoutput, LoadedNetwork, FileResultsTeLoaded);

				}

				/***********************LEARINNG AND SAVING PERCEPTRON NETWORK****************************************************************/

				else {
					NNPerceptronLearning nnp = new NNPerceptronLearning(file,R3,trainigInput,trainigOutput,validationInput,validationOutput,testingInput,testingOutput);
				}
			}

		}

	

	}
}
