using System;
using AForge.Neuro;

namespace xamarin_neural_network
{
	[Serializable]
	public class myUpdateBipolarSigmodFunction : IActivationFunction 
	{
		private double alpha = 2;
		private double bias = .250;
		private double x = 0.250;
		private int cont = 0;

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

		public myUpdateBipolarSigmodFunction(double alpha)
		{
			this.alpha = alpha;

		}

		public double Function(double x)
		{
			this.x = x;
			this.cont += 1;

			//Error err = new Error();

			//this.bias = err.biasOfx(x);

			double[][] t = NNPerceptronLearning.trainigOutput2;

			if (this.cont == t.Length) this.cont = 0;

			this.bias =  Math.Abs(t[this.cont][0] - x)/1000;

			//Console.WriteLine(x + " " + t[this.cont][0] + " " + this.bias);

			return ((x - this.bias) / this.alpha);
		}

		public double Derivative(double x)
		{
			this.x = x;
			double y = Function(x);

			Error err = new Error();
			double e = err.errorOfx(x);

			return ((y * x / this.alpha) * (e));
		}


		public double Derivative2(double y)
		{
			Error err = new Error();
			double e = err.errorOfx(this.x);

			return ((y * this.x / this.alpha) * (e));
		}


	}
}
