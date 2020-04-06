using System;
using AForge;
using Accord.Neuro.Neurons;
using Accord.Neuro.Networks;
using Accord.Neuro.Learning;
using Accord.Statistics.Distributions.Univariate;
using Accord.Math.Random;
using Accord.Math;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge.Neuro.Learning;
using System.IO;

namespace xamarin_neural_network
{
	[Serializable]

	public class myAccordSigmoidStochasticFunction : IStochasticFunction 
	{
		public  DeepBeliefNetwork net;
		public  Accord.Neuro.Learning.ParallelResilientBackpropagationLearning teacher2;
		public  mylearning teacher;

		// linear slope value
		private double alpha  = 2;
		private double reward = 0;

		//private double alpha = 2;
		private double bias    = .250;
		private double x       = 0.250;
		private int    cont    = 0;

		public double Alpha
		{
			get { return alpha; }
			set { alpha = value; }
		}

		public double Reward
		{
			get { return reward; }
			set { reward = value; }
		}

		public myAccordSigmoidStochasticFunction(double alpha)
		{
			this.alpha = alpha;
		}


		public myAccordSigmoidStochasticFunction(double alpha, DeepBeliefNetwork network, mylearning teacher2)
		{
			this.alpha = alpha;
			this.net = network;
			this.teacher = teacher2;
		}

		public myAccordSigmoidStochasticFunction(DeepBeliefNetwork network, mylearning teacher2)
			: this(2.0, network, teacher2) { }


		public myAccordSigmoidStochasticFunction(double alpha, DoubleRange range, DeepBeliefNetwork network, mylearning teacher2)
		{
			this.Alpha = alpha;
			this.net = network;
			this.teacher = teacher2;
		}




		public myAccordSigmoidStochasticFunction(double alpha, DeepBeliefNetwork network, Accord.Neuro.Learning.ParallelResilientBackpropagationLearning teacher2)
		{
			this.alpha   = alpha;
			this.net     = network;
			this.teacher2 = teacher2;
		}

		public myAccordSigmoidStochasticFunction(DeepBeliefNetwork network, Accord.Neuro.Learning.ParallelResilientBackpropagationLearning teacher2)
            : this(2.0, network, teacher2) { }

		 
		public myAccordSigmoidStochasticFunction(double alpha, DoubleRange range, DeepBeliefNetwork network, Accord.Neuro.Learning.ParallelResilientBackpropagationLearning teacher2)
		{
			this.Alpha   = alpha;
			this.net     = network;
			this.teacher2 = teacher2;
		}

		public double Function(double x)
		{
			double y=0;
			//return (1 / (1 + Math.Exp(-alpha * x))); //sigmoid

			this.x         = x;
			this.cont     += 1;

			double[][] to  = DeepLearning.trainigOutput2; //DeepLearning

			if (this.cont == to.Length) this.cont = 0;

			double error   = Math.Abs(to[this.cont][0] - x) / 1000;

			this.reward    = error;

			y = (1 / (1 + Math.Exp(-this.bias * this.bias * x))); //sigmoid

			return y;
		}

		public double Generate(double x)
		{
			// assume zero-mean noise
			//double y = (1 / (1 + Math.Exp(-alpha * x)))+ NormalDistribution.Random();

			//return y;
			/**************************************************************************/


			//Console.WriteLine(x + " " + t[this.cont][0] + " " + this.bias);

			//double y= ((this.x - this.bias) / this.alpha) + NormalDistribution.Random();
			double y = (1 / (1 + Math.Exp(-this.bias* this.bias * x))) + NormalDistribution.Random();

			return y;

		}
		 
		public double Generate2(double y)
		{
			y = y + NormalDistribution.Random();

			return y;
		}

		public double Derivative(double x)
		{
			//double y = Function(x);

			//return (alpha * y * (1 - y));
			/******************************/


			double y = Function(x);

			//return ((y * x / this.alpha) * (e));


			return (this.bias* this.bias * y * (1 - y));
		}

		public double Derivative2(double y)
		{
			//return (alpha * y * (1 - y));
			/******************************/
			/*
			Error err = new Error();
			double e = err.errorOfx(this.x);

			return ((y * this.x / this.alpha) * (e));
			*/
			return (this.bias*this.bias * y * (1 - y));
		}

	}
}
