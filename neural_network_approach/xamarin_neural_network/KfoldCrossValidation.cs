using System;
using System.Security.Cryptography;
using System.Collections.Generic;

namespace xamarin_neural_network
{

	public static class KfoldCrossValidation
	{
		private static IList<int> theList = new List<int>();

		public static IList<int> createList(int n)
		{
			IList<int> list = new List<int>();
			//list.Clear();

			for (int i = 0; i < n; i++) list.Add(i);


			return list;
		}

		public static int[] createListint(int n)
		{
			int[] listindex = new int[n];

			for (int i = 0; i < n; i++)
			{
				listindex[i] = i;

			}
			return listindex;
		}

		public static IList<T> Shuffle<T>(this IList<T> list)
		{
			theList.Clear();
			int n = list.Count;
			Random rnd = new Random();
			while (n > 1)
			{
				int k = (rnd.Next(0, n) % n);
				n--;
				T value = list[k];
				list[k] = list[n];
				list[n] = value;
				theList.Add(Convert.ToInt32(value));
			}
			return list;
		}


		public static IList<int> unsortList(IList<int> list)
		{
			list= Shuffle(list);

			//foreach (int i in list) Console.WriteLine(i);
			//Console.ReadLine();

			return list;

		}

		public static double[][] unsortedMatrix(double[][] matrix, IList<int> indexes)
		{
			double[][] unsort = new double[matrix.Length][];

			/*for (int i = 0; i < matrix.Length; i++)
			{
			    unsort[i] = matrix[indexes[i]];
			}*/

			int j = 0;
			foreach (int i in indexes)
			{
				unsort[j] = matrix[i];
				j++;
			}

			return unsort;

		}


		public static double[][] appendRowToMatrix(double[][] matrix1, double[][] matrix2, int index)
		{
			double[][] appendedMatrix = new double[matrix1.Length + 1][];

			appendedMatrix = matrix1;
			appendedMatrix[matrix1.Length] = matrix2[index];

			return appendedMatrix;
		}


		public static double[][] appendMatrixs(double[][] matrix1, double[][] matrix2)
		{
			double[][] appendedMatrix = new double[matrix1.Length + matrix2.Length][];

			for (int i = 0; i < matrix1.Length;i++)
			{
				appendedMatrix[i] = matrix1[i];
			}


			for(int j=0; j < matrix2.Length;j++)
			{
				appendedMatrix[matrix1.Length+j] = matrix2[j];
			}

			return appendedMatrix;

		}


		public static double[][] deleteMatrixLasts(double[][] matrix, int n)
		{
			double[][] deleteMatrix = new double[matrix.Length - n][];

			for (int i = 0; i < (matrix.Length - n); i++) deleteMatrix[i] = matrix[i];

			return deleteMatrix;
		}
	}
}
