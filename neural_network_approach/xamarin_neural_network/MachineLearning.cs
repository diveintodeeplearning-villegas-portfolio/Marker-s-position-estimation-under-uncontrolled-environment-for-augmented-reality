using System;
using Accord.Controls;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Math.Optimization.Losses;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Statistics.Kernels;


namespace xamarin_neural_network
{
	public class MachineLearning 
	{
		public MachineLearning(double[][] inputs, double[][] outputs2, double[][] inputs_te, double[][] outputs2_te)
		{
			Console.WriteLine("Machine learning: ");
			// We will try to model a plane as an equation in the form
			// "ax + by + c = z". We have two input variables (x and y)
			// and we will be trying to find two parameters a and b and 
			// an intercept term c.

			// We will use Ordinary Least Squares to create a
			// linear regression model with an intercept term
			var ols = new OrdinaryLeastSquares()
			{
				UseIntercept = true
			};

			// Now suppose you have some points
		/*	double[][] inputs =
			{
				new double[] { 1, 1 },
				new double[] { 0, 1 },
				new double[] { 1, 0 },
				new double[] { 0, 0 },
			};
        */

			// located in the same Z (z = 1)
			double[] outputs = new double[outputs2.Length];
			double[] outputs_te = new double[outputs2_te.Length];

			int i = 0;
			foreach (double[] o in outputs2)
			{
				outputs[i] = o[0];
				i++;
			}

			i = 0;
			foreach (double[] o in outputs2_te)
			{
				outputs_te[i] = o[0];
				i++;
			}

				// Use Ordinary Least Squares to estimate a regression model
				MultipleLinearRegression regression = ols.Learn(inputs, outputs);

			// As result, we will be given the following:
		/*	double a = regression.Weights[0]; // a = 0
			double b = regression.Weights[1]; // b = 0
			double c = regression.Weights[2]; // c = 0
			double d = regression.Weights[3]; // d = 0
			double e = regression.Intercept;  // e = 1
		*/
			// This is the plane described by the equation
			// ax + by + cz +dl+e = z => 0x + 0y + 0z +0l + 1 = z => 1 = z.

			//Write predicted values, error of training data
		  writePredictedErrors(regression, inputs, outputs);
           
			//Write predicted value, error of testing data
		  writePredictedErrors(regression, inputs_te, outputs_te);

		}

		public void writePredictedErrors(MultipleLinearRegression regression,double[][] inputs, double[] outputs)
		{

			// We can compute the predicted points using
			double[] predicted = regression.Transform(inputs);

			Console.WriteLine("Machine learning: ");
			foreach (double p in predicted) Console.WriteLine("Prediction: " + p);


			// And the squared error loss using 
			double error = new SquareLoss(outputs).Loss(predicted);
			Console.WriteLine("Square loss: " + error);


			TransformData transobj = new TransformData();


			double[] transOutput    = transobj.multiplyArrayByConst(outputs, 1000);
			double[] transPredicted = transobj.multiplyArrayByConst(predicted, 1000);


			// Mean error
			Error er = new Error();
			double absError = er.MeanAbsoluteError(transOutput, transPredicted);

			Console.WriteLine("MeanAbsoluteError:  " + absError);
			Console.WriteLine("----------------------------------------------------------------------------------------");

		}





	}
}
