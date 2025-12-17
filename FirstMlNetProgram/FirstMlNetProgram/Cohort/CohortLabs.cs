using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FirstMlNetProgram.Cohort.Cohort_Models;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Transforms.TimeSeries;
using static FirstMlNetProgram.Cohort.Cohort_Models.NiftyModels;
using static TorchSharp.torch.utils;

namespace FirstMlNetProgram.Cohort
{
    public class CohortLabs
    {
        public static void PredictNiftyUsingLags()
        {
            var mlContext = new MLContext(); //create ML context :mlcontext is like a starting point for all ML.NET operations

            string basePath = Directory.GetCurrentDirectory();  

            string projectRoot = Directory.GetParent(basePath).Parent.Parent.FullName;

            var dataPath = Path.Combine(projectRoot, "Data", "Nifty50_with_lags.csv");

            var Data = mlContext.Data.LoadFromTextFile<NiftyLagData>(dataPath, hasHeader: true, separatorChar: ',');

            var validRows = mlContext.Data.FilterRowsByMissingValues(Data, "NiftyLag1");

            var pipeline = mlContext.Transforms.Concatenate("Features",
                                                            nameof(NiftyLagData.NiftyLag1),
                                                            nameof(NiftyLagData.NiftyLag2),
                                                            nameof(NiftyLagData.NiftyLag3))                                                  
                                               .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: nameof(NiftyLagData.Nifty), 
                                                                                              featureColumnName: "Features"));
            var model = pipeline.Fit(validRows);

            var  predictionEngine = mlContext.Model.CreatePredictionEngine<NiftyLagData, NiftyPrediction>(model);

            var lastRow = mlContext.Data.CreateEnumerable<NiftyLagData>(validRows, reuseRowObject: false).Last();

            var prediction = predictionEngine.Predict(lastRow);

            Console.WriteLine($"Predicted Nifty Value: {prediction.PredictedValue}");
        }

        public static void PredictNiftyUsingLags_AutoML()
        {
            var mlContext = new MLContext(); //create ML context :mlcontext is like a starting point for all ML.NET operations

            string basePath = Directory.GetCurrentDirectory();

            string projectRoot = Directory.GetParent(basePath).Parent.Parent.FullName;

            var dataPath = Path.Combine(projectRoot, "Data", "Nifty50_with_lags.csv");

            var Data = mlContext.Data.LoadFromTextFile<NiftyLagData>(dataPath, hasHeader: true, separatorChar: ',');

            var validRows = mlContext.Data.FilterRowsByMissingValues(Data, "NiftyLag1");

            var experimentSetting = new RegressionExperimentSettings { MaxExperimentTimeInSeconds = 30 };

            var experiment = mlContext.Auto().CreateRegressionExperiment(experimentSetting);

            var result = experiment.Execute(validRows, labelColumnName: nameof(NiftyLagData.Nifty));

            var lastRow = mlContext.Data.CreateEnumerable<NiftyLagData>(validRows, reuseRowObject: false).Last();


            foreach (var run in result.RunDetails)
            {
                Console.WriteLine(" -----------------------------------------------------");
                Console.WriteLine(" Trainer Name: " + run.TrainerName);
                Console.WriteLine(" ----------- Metrics ------------------------");
                Console.WriteLine(" RSquared: " + run.ValidationMetrics.RSquared);
                Console.WriteLine(" RootMeanSquaredError: " + run.ValidationMetrics.RootMeanSquaredError);
                Console.WriteLine(" -----------------------------------------------------");
                Console.WriteLine(" predicted value: ");
                var model = run.Model;

                var  predictionEngine = mlContext.Model.CreatePredictionEngine<NiftyLagData, NiftyPrediction>(model);              

                var prediction = predictionEngine.Predict(lastRow);

                Console.WriteLine($"Predicted Nifty Value: {prediction.PredictedValue}");
            }
            Console.WriteLine(" -----------------------------------------------------");
            Console.WriteLine(" Best Model: " + result.BestRun.TrainerName);
            Console.ReadLine();
        }

        public static void PredictNiftySSA()
        {
            var mlContext = new MLContext();

            // Load data (ensure CSV has header: Price)
            string basePath = Directory.GetCurrentDirectory();

            string projectRoot = Directory.GetParent(basePath).Parent.Parent.FullName;

            var dataPath = Path.Combine(projectRoot, "Data", "Nifty50_with_lags.csv");

            var dataView = mlContext.Data.LoadFromTextFile<NiftyLagData>(dataPath, hasHeader: true, separatorChar: ',');

            // Train SSA forecaster
            int windowSize = 12;   // Look at last 12 months pattern
            int seriesLength = 120; // Total historical span window
            int trainSize = 240;   // Total rows used for training
            int horizon = 3;       // Predict next 1 month

            var pipeline = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: nameof(NiftyForecast.ForecastedNifty),
                inputColumnName: nameof(NiftyData.Nifty),
                windowSize: windowSize,
                seriesLength: seriesLength,
                trainSize: trainSize,
                horizon: horizon,
                confidenceLevel: 0.95f
            );

            var model = pipeline.Fit(dataView);

            // Create prediction engine
            var forecastEngine = model.CreateTimeSeriesEngine<NiftyData, NiftyForecast>(mlContext);

            var result = forecastEngine.Predict();

            for (int i = 0; i < result.ForecastedNifty.Length; i++)
            {
                Console.WriteLine($"Month +{i + 1}: {result.ForecastedNifty[i]:N2}");
            }

            Console.ReadLine();
        }
    }
}
