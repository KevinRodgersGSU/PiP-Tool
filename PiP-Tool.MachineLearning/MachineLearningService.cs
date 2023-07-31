using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML;
using PiP_Tool.MachineLearning.DataModel;
using PiP_Tool.Shared;

namespace PiP_Tool.MachineLearning
{
    public class MachineLearningService
    {

        #region public

        public static MachineLearningService Instance => _instance ?? (_instance = new MachineLearningService());

        public bool DataExist => Directory.Exists(Constants.FolderPath) && File.Exists(Constants.DataPath);
        public bool ModelExist => Directory.Exists(Constants.FolderPath) && File.Exists(Constants.ModelPath);

        #endregion

        #region private


        private static MachineLearningService _instance;
        private ITransformer _model;
        private readonly TaskCompletionSource<bool> _ready;
        private readonly SemaphoreSlim _semaphore;
        private MLContext _mlContext;

        #endregion

        private MachineLearningService()
        {
            Logger.Instance.Info("ML : Init machine learning service");

            _semaphore = new SemaphoreSlim(1);
            _ready = new TaskCompletionSource<bool>();
            _mlContext = new MLContext();

            if (!Directory.Exists(Constants.FolderPath))
                Directory.CreateDirectory(Constants.FolderPath);
        }

        public void Init()
        {
            if (_ready.Task.IsCompleted)
                return;

            Task.Run(async () =>
            {
                if (!ModelExist)
                    await Train();
                else
                    try
                    {
                        _model = _mlContext.Model.Load(Constants.ModelPath, out var modelSchema);
                    }
                    catch (Exception)
                    {
                        File.Delete(Constants.ModelPath);
                        await Train();
                    }
            }).ContinueWith(obj =>
            {
                _ready.SetResult(true);
            });
        }
        ~MachineLearningService() => Dispose();
        public void Dispose()
        {
        }

        public async Task TrainAsync()
        {
            if (!_ready.Task.IsCompleted)
                await _ready.Task;

            await Train();
        }
        private async Task Train()
        {
            try
            {
                Logger.Instance.Info("ML : Training model");
                CheckDataFile();
                var trainingDataView = _mlContext.Data.LoadFromTextFile<WindowData>(Constants.DataPath, separatorChar: ',');
                var pipeline = _mlContext.Transforms.Text.FeaturizeText(outputColumnName: "PredictedLabel", inputColumnName: "Label")
                    .Append(_mlContext.Transforms.Text.FeaturizeText(outputColumnName: "ProgramFeaturized", inputColumnName: nameof(WindowData.Program)))
                    .Append(_mlContext.Transforms.Text.FeaturizeText(outputColumnName: "WindowTitleFeaturized", inputColumnName: nameof(WindowData.WindowTitle)))
                    .Append(_mlContext.Transforms.Concatenate("Features", "ProgramFeaturized", "WindowTitleFeaturized"))
                    .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "WindowTop", featureColumnName: "Features"))
                    .Append(_mlContext.Transforms.Conversion.MapValueToKey("PredictedLabel"));
                await _semaphore.WaitAsync();
                _model = pipeline.Fit(trainingDataView);
                _semaphore.Release();
                _mlContext.Model.Save(_model, trainingDataView.Schema, Constants.ModelPath);
                Logger.Instance.Info("ML : Model trained");
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
            }
        }
        public async Task<RegionPrediction> PredictAsync(string program, string windowTitle, float windowTop, float windowLeft, float windowHeight, float windowWidth)
        {
            return await PredictAsync(new WindowData
            {
                Program = program,
                WindowTitle = windowTitle,
                WindowTop = windowTop,
                WindowLeft = windowLeft,
                WindowHeight = windowHeight,
                WindowWidth = windowWidth
            });
        }
        public async Task<RegionPrediction> PredictAsync(WindowData windowData)
        {
            if (!_ready.Task.IsCompleted)
                await _ready.Task;

            //await _semaphore.WaitAsync();
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<WindowData, RegionPrediction>(_model);
            var prediction = predictionEngine.Predict(windowData);
            _semaphore.Release();

            prediction.Predicted();

            Logger.Instance.Info("ML : Predicted : " + prediction + " From " + windowData);

            return prediction;
        }

        public void AddData(string region, string program, string windowTitle, float windowTop, float windowLeft, float windowHeight, float windowWidth)
        {
            Logger.Instance.Info("ML : Add new data");

            var newLine =
                $"{Environment.NewLine}" +
                $"{region}," +
                $"{program}," +
                $"{windowTitle}," +
                $"{windowTop}," +
                $"{windowLeft}," +
                $"{windowHeight}," +
                $"{windowWidth}";

            if (!File.Exists(Constants.DataPath))
                File.WriteAllText(Constants.DataPath, "");

            File.AppendAllText(Constants.DataPath, newLine);
        }

        private void CheckDataFile()
        {
            if (!DataExist)
            {
                Logger.Instance.Warn("ML : " + DataExist + " doesn't exist");
                File.WriteAllText(Constants.DataPath, "");
            }
            var lineCount = File.ReadLines(Constants.DataPath).Count();
            if (lineCount >= 3)
                return;
            Logger.Instance.Warn("ML : No or not enough data");
            AddData("0 0 100 100", "PiP", "PiP", 0, 0, 100, 100);
            AddData("0 0 100 100", "Tool", "Tool", 0, 0, 200, 200);
            AddData("100 100 200 200", "Test", "Test", 0, 0, 300, 300);
        }
    }
}
