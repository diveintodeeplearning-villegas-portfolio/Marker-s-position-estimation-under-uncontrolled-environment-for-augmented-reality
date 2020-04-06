using System;
using AForge;
using AForge.Math.Random;
using Accord.Neuro.Neurons;
using Accord.Neuro.Networks;
using Accord.Neuro.Learning;
using Accord.Statistics.Distributions.Univariate;

namespace xamarin_neural_network
{
	[Serializable]
	public class myGaussianFunction : Accord.Neuro.ActivationFunctions.IStochasticFunction
	{
		
		// linear slope value
		private double alpha = 1;

		// function output range
		private DoubleRange range = new DoubleRange(-1, +1);

		private static StandardGenerator gaussian = new StandardGenerator(Environment.TickCount);

		/// <summary>
		///   Gets or sets the class-wide  
		///   Gaussian random generator.
		/// </summary>
		/// 
		public static StandardGenerator Random
		{
			get { return gaussian; }
			set { gaussian = value; }
		}

		/// <summary>
		/// Linear slope value.
		/// </summary>
		/// 
		/// <remarks>
		///   <para>Default value is set to <b>1</b>.</para>
		/// </remarks>
		/// 
		public double Alpha
		{
			get { return alpha; }
			set { alpha = value; }
		}

		/// <summary>
		///   Function output range.
		/// </summary>
		///
		/// <remarks>
		///   <para>Default value is set to [-1;+1]</para>
		/// </remarks>
		///
		public DoubleRange Range
		{
			get { return range; }
			set { range = value; }
		}

		/// <summary>
		///   Creates a new <see cref="GaussianFunction"/>.
		/// </summary>
		/// 
		/// <param name="alpha">The linear slope value. Default is 1.</param>
		/// 
		public myGaussianFunction(double alpha)
		{
			this.alpha = alpha;
		}

		/// <summary>
		///   Creates a new <see cref="GaussianFunction"/>.
		/// </summary>
		/// 
		public myGaussianFunction()
            : this(1.0) { }

		/// <summary>
		///   Initializes a new instance of the <see cref="GaussianFunction"/> class.
		/// </summary>
		/// 
		public myGaussianFunction(double alpha, DoubleRange range)
		{
			this.Alpha = alpha;
			this.Range = range;
		}


		/// <summary>
		/// Calculates function value.
		/// </summary>
		///
		/// <param name="x">Function input value.</param>
		/// 
		/// <returns>Function output value, <i>f(x)</i>.</returns>
		///
		/// <remarks>The method calculates function value at point <paramref name="x"/>.</remarks>
		///
		public double Function(double x)
		{
			double y = alpha * x;

			if (y > range.Max)
				return range.Max;
			else if (y < range.Min)
				return range.Min;
			return y;
		}

		/// <summary>
		///   Samples a value from the function given a input value.
		/// </summary>
		/// 
		/// <param name="x">Function input value.</param>
		/// 
		/// <returns>
		///   Draws a random value from the function.
		/// </returns>
		/// 
		public double Generate(double x)
		{
			// assume zero-mean noise
			double y = alpha * x + gaussian.Next();

			if (y > range.Max)
				y = range.Max;
			else if (y < range.Min)
				y = range.Min;

			return y;
		}

		/// <summary>
		///   Samples a value from the function given a function output value.
		/// </summary>
		/// 
		/// <param name="y">Function output value - the value, which was obtained
		/// with the help of <see cref="Function"/> method.</param>
		/// 
		/// <returns>
		///   Draws a random value from the function.
		/// </returns>
		/// 
		public double Generate2(double y)
		{
			y = y + gaussian.Next();

			if (y > range.Max)
				y = range.Max;
			else if (y < range.Min)
				y = range.Min;

			return y;
		}

		/// <summary>
		/// Calculates function derivative.
		/// </summary>
		/// 
		/// <param name="x">Function input value.</param>
		/// 
		/// <returns>Function derivative, <i>f'(x)</i>.</returns>
		/// 
		/// <remarks>The method calculates function derivative at point <paramref name="x"/>.</remarks>
		///
		public double Derivative(double x)
		{
			double y = alpha * x;

			if (y <= range.Min || y >= range.Max)
				return 0;
			return alpha;
		}

		/// <summary>
		/// Calculates function derivative.
		/// </summary>
		/// 
		/// <param name="y">Function output value - the value, which was obtained
		/// with the help of <see cref="Function"/> method.</param>
		/// 
		/// <returns>Function derivative, <i>f'(x)</i>.</returns>
		/// 
		/// <remarks><para>The method calculates the same derivative value as the
		/// <see cref="Derivative"/> method, but it takes not the input <b>x</b> value
		/// itself, but the function value, which was calculated previously with
		/// the help of <see cref="Function"/> method.</para>
		/// 
		/// <para><note>Some applications require as function value, as derivative value,
		/// so they can save the amount of calculations using this method to calculate derivative.</note></para>
		/// </remarks>
		/// 
		public double Derivative2(double y)
		{
			if (y <= range.Min || y >= range.Max)
				return 0;
			return alpha;
		}

	}
}
