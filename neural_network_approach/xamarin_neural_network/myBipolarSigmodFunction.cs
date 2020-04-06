using System;
using AForge.Neuro;

namespace xamarin_neural_network 
{
	[Serializable]
	public class myBipolarSigmodFunction : IActivationFunction
	{
		private double alpha = 2;
		private double bias = .250;
		private double x = 0.250;

		public double Alpha
		{
			get { return alpha; }
			set { alpha = value; }
		}

		public double Bias
		{
			get { return bias; }
			set { bias = value; }
		}

		public double X
		{
			get { return x; }
			set { x = value; }
		}

		public myBipolarSigmodFunction(double alpha, double bias)
		{
			this.alpha = alpha;
			this.bias  = bias;
		}

		public double Function(double x)
		{
			this.x = x;
			return ((x-this.bias)/this.alpha);
		}

		public double Derivative(double x)
		{
			this.x    = x;
			double y  = Function(x);

			Error err = new Error();
			double e  = err.errorOfx(x);

			return ((y*x/this.alpha)*(e));
		}


		public double Derivative2(double y)
		{
			Error err = new Error();
			double e = err.errorOfx(this.x);

			return ((y * this.x / this.alpha) * (e));
		}


	}
}
