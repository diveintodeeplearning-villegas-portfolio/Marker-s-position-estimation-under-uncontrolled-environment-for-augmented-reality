using System;
namespace xamarin_neural_network
{
	public class TransformData
	{
		public TransformData()
		{
		
	    }
	
		public String[][] doubleArrayToString(double[][] doubleArray) 
		{
			int j = 0;
			String[][] StrArray = { new string[] { "0" }, new string[] { "0" }, new string[] { "0" } , new string[] { "0" }, new string[] { "0" },
									new string[] { "0" }, new string[] { "0" }, new string[] { "0" } , new string[] { "0" }, new string[] { "0" },
									new string[] { "0" }, new string[] { "0" }, new string[] { "0" } , new string[] { "0" }, new string[] { "0" },
									new string[] { "0" }, new string[] { "0" }, new string[] { "0" } , new string[] { "0" }, new string[] { "0" },
									new string[] { "0" }, new string[] { "0" }, new string[] { "0" } , new string[] { "0" }, new string[] { "0" },
									new string[] { "0" }, new string[] { "0" }, new string[] { "0" } , new string[] { "0" }, new string[] { "0" },
									new string[] { "0" }, new string[] { "0" }, new string[] { "0" } , new string[] { "0" }, new string[] { "0" },
									new string[] { "0" }, new string[] { "0" }, new string[] { "0" } , new string[] { "0" }, new string[] { "0" },
									new string[] { "0" }
									};

			foreach (double[] n in doubleArray)
			{
				StrArray[j] = Array.ConvertAll<double, string>(n, Convert.ToString);
				j++;
			}
			return StrArray;
			
		}
		/*********************** Multiply Vector of a Matrix by a Constant *****************************************************************/
		public double[] multiplyVectorByConst(double[][] XArgs, int col, double c)
		{
			double[] array=new double [XArgs.Length];

			for (int i = 0; i < XArgs.Length; i++) array[i] = XArgs[i][col] * c;
			
			return array;
		}

		public double[] multiplyArrayByConst(double[] XArgs, double c)
		{
			double[] array = new double[XArgs.Length];

			for (int i = 0; i < XArgs.Length; i++) array[i] = XArgs[i] * c;

			return array;
		}

		/***********************TRANSFOR ARRAY TO LN *****************************************************************/
		public double[] transToLn(double[] XArgs){
			double[] transData = XArgs;
			int i = 0;

			Console.WriteLine("  Evaluate this identity with selected values for X:");
			Console.WriteLine("                              ln(x) = 1 / log[X](B)");
			Console.WriteLine();

			foreach (double argX in XArgs)
			{
				// Find natural log of argX.
				Console.WriteLine("                      Math.Log({0}) = {1:E16}",
								  argX, Math.Log(argX));

				// Evaluate 1 / log[X](e).
				Console.WriteLine("             1.0 / Math.Log(e, {0}) = {1:E16}",
								  argX, 1.0 / Math.Log(Math.E, argX));
				Console.WriteLine();

				transData[i] = 1.0 / Math.Log(Math.E, argX);
				i++;
			}

			return transData;
		}

		public double transToLn(double XArgs)
		{
			double transData = XArgs;

			transData = 1.0 / Math.Log(Math.E, XArgs);

			return transData;
		}
		/***********************TRANSFORM MATRIX TO LN****************************************************************/
		public double[][] transMatrixToLn(double[][] MArgs) {
			   double[][] transMatrix = MArgs;

			for (int i = 0; i < MArgs.Length; i++)
				transMatrix[i]=transToLn(transMatrix[i]);
			return transMatrix;
		}

		/***********************TRANSFORM A MATRIX TO X/100 y X/1000 *****************************************************************/

		public double[] transVectorToDecimalInput(double[] Array)
		{
			int i = 0;
			foreach (double x in Array)
			{

				Array[i] = Math.Abs(x) / 1000;

				i++;
			}
			return Array;

		}

		/***********************TRANSFORM A MATRIX TO X/1000 *****************************************************************/

		public double[] transVectorToDecimalOutput(double[] Array)
		{
			int i = 0;
			foreach (double x in Array)
			{
				
				Array[i] = Math.Abs(x) / 1000;

				i++;
			}
			return Array;

		}
	
		/***********************TRANSFORM MATRIX TO X/1000 *****************************************************************/

		public double[][] transMatrixToDecimalInput(double[][] Matrix)
		{
			double[][] m = Matrix;
			int i = 0;

			foreach (double[] x in Matrix)
			{
				m[i] = transVectorToDecimalInput(x);
				i++;
			}

			return m;

		}

		/***********************TRANSFORM MATRIX TO X/1000 *****************************************************************/

		public double[][] transMatrixToDecimalOutput(double[][] Matrix)
		{
			double[][] m = Matrix;
			int i = 0;

			foreach (double[] x in Matrix)
			{
				m[i] = transVectorToDecimalOutput(x);
				i++;
			}

			return m;

		}
		/***********************TRANSFORM A COLUM OF A MATRIX TO X/1000 *****************************************************************/


		public double[][] transMatrixToDecimal(double[][] Matrix, int col) //1 2 3 4
		{
			int j = col;
			//int rows = myArray.GetLength(0);
			//int columns = myArray.GetLength(1);

			for (int i = 0; i < Matrix.Length; i++)
			{

				Matrix[i][j] = Math.Abs(Matrix[i][j]) / 1000;
				//Console.WriteLine("output: "+output[i][0]);

			}
			//Console.ReadLine();
			return Matrix;
		}



	}
}
