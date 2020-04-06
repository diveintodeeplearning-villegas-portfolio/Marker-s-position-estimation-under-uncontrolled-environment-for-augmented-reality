using System;
namespace xamarin_neural_network
{
	using System;
	using System.Threading;
	using System.Threading.Tasks;
	using Accord.Neuro;
	using AForge.Neuro.Learning;
	using Accord.Neuro.ActivationFunctions;
	using Accord.Neuro.Learning;
	using Accord.Neuro.Networks;
	using Accord.Math;
	using System.Collections.Generic;
	using System.Linq;
	using System.Text;
	using System.IO;

	public class Action
	{
		public List<double>  hiddenWeights;

		public List<double>  visibleWeights;

		public List<double>  hiddenTreshold;

		public List<double>  visibleTreshold;

		public double        sumOfSquaredErrors;

		public List<double>  HiddenWeights
		{
			get { return hiddenWeights; }
			set { hiddenWeights = value; }
		}

		public List<double> VisibleWeights
		{
			get { return visibleWeights; }
			set { visibleWeights = value; }
		}

		public List<double> HiddenTreshold
		{
			get { return hiddenTreshold; }
			set { hiddenTreshold = value; }
		}

		public List<double> VisibleTreshold
		{
			get { return visibleTreshold; }
			set { visibleTreshold = value; }
		}


		public double SumOfSquaredErrors
		{
			get { return sumOfSquaredErrors; }
			set { sumOfSquaredErrors = value; }
		}

		public Action()
		{
			 //this.hiddenWeights.Clear();
			 //this.visibleWeights.Clear();
			 this.sumOfSquaredErrors = 0;
		}


	}
}
