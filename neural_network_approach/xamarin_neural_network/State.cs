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

	public class State
	{
		public List<double> hiddenWeights;

		public List<double> visibleWeights;

		public List<double> hiddenTreshold;

		public List<double> visibleTreshold;

		public double reward;

		public int    epochs;

		public List<double> HiddenWeights
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


		public double Reward
		{
			get { return reward; }
			set { reward = value; }
		}

		public int Epochs
		{
			get { return epochs; }
			set { epochs = value; }
		}

		public State()
		{
			this.Reward  = 10000000;
			this.epochs  = 1;
		}
	}
}
