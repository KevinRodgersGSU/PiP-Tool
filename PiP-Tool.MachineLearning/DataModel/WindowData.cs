using Microsoft.ML.Data;

namespace PiP_Tool.MachineLearning.DataModel
{
    public class WindowData
    {

        /// <summary>
        /// Labal: predicted region (format: "Top Left Height Width")
        /// </summary>
        [LoadColumn(0), ColumnName("Label")]
        public string Region;

        /// <summary>
        /// Name of the program
        /// </summary>
        [LoadColumn(1), ColumnName("Program")]
        public string Program;

        /// <summary>
        /// Title of the window
        /// </summary>
        [LoadColumn(2), ColumnName("WindowTitle")]
        public string WindowTitle;

        /// <summary>
        /// Top position of the window
        /// </summary>
        [LoadColumn(3), ColumnName("WindowTop")]
        public float WindowTop;

        /// <summary>
        /// Left position of the window
        /// </summary>
        [LoadColumn(4), ColumnName("WindowLeft")]
        public float WindowLeft;

        /// <summary>
        /// Height of the window
        /// </summary>
        [LoadColumn(5), ColumnName("WindowHeight")]
        public float WindowHeight;

        /// <summary>
        /// Width of the window
        /// </summary>
        [LoadColumn(6), ColumnName("WindowWidth")]
        public float WindowWidth;
        
        public override string ToString()
        {
            return "WindowTitle : " + WindowTitle + ", " +
                   "WindowTop : " + WindowTop + ", " +
                   "WindowLeft : " + WindowLeft + ", " +
                   "WindowHeight : " + WindowHeight + ", " +
                   "WindowWidth : " + WindowWidth;
        }

    }
}

