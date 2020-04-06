using System;
using AForge.Neuro;



namespace xamarin_neural_network
{
	[Serializable]
	public class myActivationFunction : IActivationFunction
	{
		

		public myActivationFunction()
		{
		}


		public double Function(double x)
		{
			return Normaldistr.normaldistribution(x);
		}

		public double Derivative(double x)
		{
			double y = Function(x);


			return (-2*y*x);
		}


		public double Derivative2(double y)
		{
			double x = 0;

			TransformData trans = new TransformData();

			//(ln(1/((2/pi^2*exp(-t^2))/(2/pi^2))))^0.5
			//(ln(1/y/(2/pi^2))))^0.5

			double c = 2/Math.Pow(Math.PI, 2);

			x =Math.Pow(trans.transToLn((1/y/c)),0.5);

			return (-2 * y * x );
		}



	}
}