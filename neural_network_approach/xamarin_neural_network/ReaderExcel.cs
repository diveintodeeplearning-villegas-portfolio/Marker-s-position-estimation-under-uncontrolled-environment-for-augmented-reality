using System;
using System.IO;
using Excel;
using ICSharpCode.SharpZipLib;
using ICSharpCode;
using System.Data;
using System.Collections.Generic;
using Microsoft.VisualBasic.FileIO;




namespace xamarin_neural_network
{
	public class ReaderExcel
	{
		/***********************GLOBAL VARIABLE IN READER EXCEL CLASS*****************************************************************/

		public List<String[]> fileContent = new List<string[]>();
		public string[][] line= new string [920][]; //Kinect 41 //Hearta1 920

	
		/***********************READING CSV*****************************************************************/
		public ReaderExcel()
		{ 
		}

		public ReaderExcel(String FileName, String Delimiters)
		{
			int i = 0;

			//List<String[]> fileContent = new List<string[]>();

			using (FileStream reader      = File.OpenRead(FileName)) // mind the encoding - UTF8
			using (TextFieldParser parser = new TextFieldParser(reader))
			{
				parser.TrimWhiteSpace = true; // if you want
				if(Delimiters =="Comma")
					parser.Delimiters = new[] { "," }; // { "," }; when it's separated by cells //{ " " }; when are in a single line all values
				else
					parser.Delimiters     = new[] { " " }; // { "," }; when it's separated by cells //{ " " }; when are in a single line all values

				parser.HasFieldsEnclosedInQuotes = true;
				while (!parser.EndOfData)
				{
					line[i] = parser.ReadFields();
					fileContent.Add(line[i]);
					//Console.WriteLine(line[i][0]+line[i][1]+line[i][2]+line[i][3]+line[i][4]+line[i][5]+line[i][6]);
					i++;
				}
				line = fileContent.ToArray();
			}
			//Console.ReadLine();


		}

		/***********************GET DATA TO DOUBLE *****************************************************************/
		public double[][] getdata()
		{
			double[][] array = new double[fileContent.Count][];

			for (int i = 0; i < fileContent.Count; i++)
			{
				array[i]= Array.ConvertAll<string, double>(line[i], Convert.ToDouble);
				//Console.WriteLine(array[i][0]+ "--"+array[i][1]+"--"+array[i][2]+"--"+array[i][3]+"--"+array[i][4]+"--"+array[i][5]+"--"+array[i][6]);
			}
			//Console.ReadLine();
			return array;
		}

		/***********************GET INPUT FROM DATA****************************************************************/
		public double[][] getInput()
		{
			double[][] input = new double[fileContent.Count][];
			double[][] data = getdata();

			for (int i = 0; i < data.Length; i++)
			{
				//for (int j = 3; j < 7; j++)
				//{
					//	double[] XArgs1 = { 10, 2 };
				input[i] = new double[] { Math.Abs(data[i][3]),Math.Abs(data[i][4]),Math.Abs(data[i][5]),Math.Abs(data[i][6]) };
					//Console.WriteLine(input[i][0] + "--" + input[i][1] + "--" + input[i][2] + "--" + input[i][3]);

				//}
			}
			//Console.ReadLine();

			return input;

		}

		/***********************GET INPUT FROM DATA****************************************************************/
		public double[][] getInput(int numInput)
		{
			double[][] input = new double[fileContent.Count][];
			double[][] data = getdata();


			for (int i = 0; i < data.Length; i++)
			{


				input[i] = new double[] {
					       Math.Abs(data[i][0]),  Math.Abs(data[i][1]),  Math.Abs(data[i][2]), 
					       Math.Abs(data[i][3]),  Math.Abs(data[i][4]),  Math.Abs(data[i][5]),  Math.Abs(data[i][6]), 
						   Math.Abs(data[i][7]),  Math.Abs(data[i][8]),  Math.Abs(data[i][9]),  Math.Abs(data[i][10]),
						   Math.Abs(data[i][11]), Math.Abs(data[i][12]), Math.Abs(data[i][13]), Math.Abs(data[i][14]),
					 	   Math.Abs(data[i][15]), Math.Abs(data[i][16]), Math.Abs(data[i][17]), Math.Abs(data[i][18]),
						   Math.Abs(data[i][19]), Math.Abs(data[i][20]), Math.Abs(data[i][21]), Math.Abs(data[i][22]),
						   Math.Abs(data[i][23]), Math.Abs(data[i][24]), Math.Abs(data[i][25]), Math.Abs(data[i][26]),
					       Math.Abs(data[i][27]), Math.Abs(data[i][28]), Math.Abs(data[i][29]), Math.Abs(data[i][30]),
					       Math.Abs(data[i][31]), Math.Abs(data[i][32]), Math.Abs(data[i][33]), Math.Abs(data[i][34])				        
				};

			}
			//Console.ReadLine();

			return input;

		}
		/***********************GET OUTPUT FROM DATA*****************************************************************/
		public double[][] getOutput(int indexOutput)
		{
			double[][] output = new double[fileContent.Count][];
			double[][] data = getdata();

			for (int i = 0; i < data.Length; i++)
			{
				//for (int j = 2; j < 3; j++)
				//{
				output[i] = new double[] { Math.Abs(data[i][indexOutput]) };
				//Console.WriteLine("output: "+output[i][0]);

				//}
			}
			//Console.ReadLine();

			return output;
		}

		/***********************GET INDEX OF RANGE***************************************************************/
		public int[] getIndexRange(LightRange lightrange)
		{
			List<int> indexRange = new List<int>();

			double[][] data = getdata();
			int i = 0;

				foreach (double[] r in data)
				{
					if (r[6] >= lightrange.getLightMin() && r[6] < lightrange.getLightMax()) 
					{
						indexRange.Add(i);
					}
					i++;
				}

			return indexRange.ToArray();
		}
		/***********************GET RANGED INPUT****************************************************************/

		public double[][] getRangeIntput(LightRange range)
		{
			int   []    indexsRange = getIndexRange(range);
			double[][]  rangeInput  = new double[indexsRange.Length][];
			double[][]  input       = getInput();
			int                   i = 0;

			foreach (int index in indexsRange)
			{
				if(i>=0) rangeInput[i] = input[index];
				i++;
			}

			return rangeInput;
		}
		/***********************GET RANGED OUTPUT****************************************************************/

		public double[][] getRangeOutput(LightRange range)
		{
			int[]      indexsRange = getIndexRange(range);
			double[][] rangeOutput = new double[indexsRange.Length][];
			double[][] output      = getOutput();
			int                  i = 0;

			foreach (int index in indexsRange)
			{
				rangeOutput[i] = output[index];
				i++;
			}

			return rangeOutput;
		}

		/***********************GET OUTPUT FROM DATA*****************************************************************/
		public double[][] getOutput()
		{
			double[][] output = new double[fileContent.Count][];
			double[][] data = getdata();

			for (int i = 0; i < data.Length; i++)
			{
				//for (int j = 2; j < 3; j++)
				//{
				output[i] = new double[] { Math.Abs(data[i][2]) };
				//Console.WriteLine("output: "+output[i][0]);

				//}
			}
			//Console.ReadLine();


			return output;
		}


		/***********************WRITING CSV******************************/
		public void WriterExcel(String FileName, string[][] results)
		{

			using (FileStream fileStream = new FileStream(FileName, FileMode.Append, FileAccess.Write))
			using (var outputFile = new StreamWriter(fileStream))
			{
				foreach (string[] r in results) outputFile.WriteLine(r[0]);
			}

		}
	}
}
