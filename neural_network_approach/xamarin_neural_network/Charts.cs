using System;
namespace xamarin_neural_network
{
	public class Charts //: UIViewController, INChartSeriesDataSource
	{
		public Charts() 
		{
			
		}
	/*	public override void LoadView()
		{
			// Create a chart view that will display the chart.
			View = new NChartView();
			// Paste your license key here.
			view.Chart.LicenseKey = "";
			// Create series.
			NChartColumnSeries series = new NChartColumnSeries();
			series.Brush = NChartSolidColorBrush.SolidColorBrushWithColor(UIColor.Red);
			// Set data source.
			series.DataSource = this;
			// Add series to the chart.
			view.Chart.AddSeries(series);
			// Update data in the chart.
			view.Chart.UpdateData();
			// Set chart view to the controller.
			this.View = View;
		}
		// Get points for the series.
		public NChartPoint[] SeriesDataSourcePointsForSeries(NChartSeries series)
		{
			// Create points with some data for the series.
			List result = new List();
			for (int i = 0; i <= 10; ++i)
			{
				// You can use any custom data source. For example, use (new Random()).Next(30)
				NChartPointState state = NChartPointState.PointStateAlignedToXWithXY(i, Core.MyDataSource.NextValue(30));
				result.Add(NChartPoint.PointWithState(state, series));
			}
			return result.ToArray();
		}
		// Get name of the series.
		public string SeriesDataSourceNameForSeries(NChartSeries series)
		{
			return "My series";
		}
		// If you don't need to customize bitmap in the series, return null.
		public UIImage SeriesDataSourceImageForSeries(NChartSeries series) { return null; }
*/
}

}
