using System;
namespace xamarin_neural_network
{
	public class Error
	{
		public Error()
		{
			//Do nothing
		}

		public double errorOfx(double x)
		{
			double e = 0;

			TransformData TransData = new TransformData();
			ReaderExcel reader = new ReaderExcel();

			double[][] input = reader.getInput();
			double[][] output = reader.getdata();

			input = TransData.transMatrixToDecimalInput(input); //TransData.transMatrixToLn(theinput);
			output = TransData.transMatrixToDecimalOutput(output); //TransData.transMatrixToDecimal(theoutput,0)

			int j = 0;
			foreach (double[] i in input)
			{
				if (x == i[2]) e=Math.Abs(i[2]-output[j][0]);
				j++;
			}


			return e;
		}

		public double biasOfx(double x)
		{
			double bias = 0.250;

			ReaderExcel reader = new ReaderExcel();
			TransformData TransData = new TransformData();

			double[][] input = reader.getInput();
			double[][] output = reader.getdata();
			input = TransData.transMatrixToDecimalInput(input); //TransData.transMatrixToLn(theinput);
			output = TransData.transMatrixToDecimalOutput(output); //TransData.transMatrixToDecimal(theoutput,0)


			int j = 0;
			foreach (double[] i in input)
			{
				if (x == i[2]) bias = output[j][0];
				j++;
			}

			return bias;
		}

		public double MeanAbsoluteError(double[] y, double[] f)
		{
			double sumerror = 0;
			double meanerror = 0;

			for (int i = 0; i < y.Length; i++)
			{
				sumerror += Math.Abs(y[i] - f[i]);

			}
			meanerror = sumerror / y.Length;

			return meanerror;

		}

		public double Mean(double[] y)
		{
			double suma = 0;
			double mean = 0;

			foreach (double yi in y) suma += yi;

			mean = suma / y.Length;

			return mean;

		}


		public double TotalSumSquares(double[] y, double ymean)
		{
			double ssq = 0;

			foreach (double yi in y) ssq += Math.Sqrt(Math.Abs(yi - ymean));


			return ssq;

		}


		public double SquareResidual(double[] y, double[] f)
		{
			double sr = 0;

			for (int i = 0; i < y.Length; i++) sr += Math.Sqrt(Math.Abs(y[i] - f[i]));

			return sr;
		}

		public double squareR(double[] y, double[] f, double ymean)
		{
			double sr = 0;

			sr = 1 - SquareResidual(y, f) / TotalSumSquares(y, ymean);

			return sr;
		}
	}
}
