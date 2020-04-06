using System;
using AForge.Neuro;



namespace xamarin_neural_network
{
	[Serializable]
	public class myRegressionFunction : IActivationFunction
	{


		public myRegressionFunction()
		{
		}


		public double Function(double x)
		{
			
			return x;
		}

		public double Derivative(double x)
		{
			double y = Function(x);


			return (-2 * y * x);
		}


		public double Derivative2(double y)
		{
			double x = y;


			return (-2 * y * x);
		}



	}
}