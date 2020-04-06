using System;
namespace xamarin_neural_network
{
	public class LightRange
	{
		public int min;
		public int max;
		public int index;

		public LightRange()
		{
			index = 0;
			min   = 0;
			max   = 0;
		}
		public LightRange(int idx, int minimum, int maximum)
		{
			setLightRange(idx, minimum, maximum);
		}

		public void setLightRange(int idx, int minimum, int maximum)
		{
			index = idx;
			min   = minimum;
			max   = maximum;
		}
		public int[] getLightRange()
		{
			int[] range =new int[3];

			range[0] = index;
		    range[1] = min;
			range[2] = max;

			return range; 
		}
		public int getLightMin()
		{
			return min;
		}
		public int getLightMax()
		{
			return max;
		}
		public int getLightIndex()
		{
			return index;
		}

	}
}
