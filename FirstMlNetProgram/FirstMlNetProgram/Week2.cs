using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FirstMlNetProgram.Models;
using FirstMlNetProgram.Data;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace FirstMlNetProgram
{
    public class Week2
    {
        public static void Lab6_SimplestMLAutoMl_With_HugrData()
        {
            var mlcontext = new MLContext();

            string filePath = "C:\\Users\\prach\\source\\repos\\Ml.Net\\SampleData\\linear_insurance_100k.csv";

            var data = mlcontext.Data.LoadFromTextFile<InsuranceData>(filePath, hasHeader: true, separatorChar: ',');

            var splitData = mlcontext.Data.TrainTestSplit(data, testFraction: 0.2);

            var trainData = splitData.TrainSet;

            var testData = splitData.TestSet;

            var experimentSetting = new RegressionExperimentSettings { MaxExperimentTimeInSeconds = 30 };
            var experiment = mlcontext.Auto().CreateRegressionExperiment(experimentSetting);

            var result = experiment.Execute(data, labelColumnName: "Premium");            

            foreach (var run in result.RunDetails)
            {
                Console.WriteLine(" -----------------------------------------------------");
                Console.WriteLine(" Trainer Name: " + run.TrainerName);
                Console.WriteLine(" ----------- Metrics ------------------------");
                Console.WriteLine(" RSquared: " + run.ValidationMetrics.RSquared);
                Console.WriteLine(" RootMeanSquaredError: " + run.ValidationMetrics.RootMeanSquaredError);
            }
            Console.WriteLine("************************************************");

            Console.WriteLine(" Best Model: " + result.BestRun.TrainerName);
            Console.WriteLine(" Best Model RSquared: " + result.BestRun.ValidationMetrics.RSquared);
            Console.WriteLine(" Best Model RootMeanSquaredError: " + result.BestRun.ValidationMetrics.RootMeanSquaredError);
            Console.WriteLine(" Best Model RuntimeInSeconds: " + result.BestRun.RuntimeInSeconds);

            Console.WriteLine("************************************************");

            var prdictions = mlcontext.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(result.BestRun.Model);

            var pedictedData = prdictions.Predict(new InsuranceData() { Age = 55 });

            Console.WriteLine(" Predicted Premium for Age 56 : " + pedictedData.PredictedPremium);

            Console.ReadLine();
        }

        public static void Lab6_LargeFileTesting_withAutoMLOutput()
        {
            var mlcontext = new MLContext();

            string filePath = "C:\\Users\\prach\\source\\repos\\Ml.Net\\SampleData\\linear_insurance_100k.csv";

            var data = mlcontext.Data.LoadFromTextFile<InsuranceData>(filePath, hasHeader: true, separatorChar: ',');

            var splitData = mlcontext.Data.TrainTestSplit(data, testFraction: 0.2);

            var trainData = splitData.TrainSet;

            var testData = splitData.TestSet;

            var pipeline = mlcontext.Transforms.Concatenate("f1", "Age")
                                               .Append(mlcontext.Regression
                                               .Trainers.Ols(labelColumnName: "Premium", featureColumnName: "f1"));
            var model = pipeline.Fit(trainData);

            var pedictions = mlcontext.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);

            var predictedData = pedictions.Predict(new InsuranceData() { Age = 68 });

            Console.Write(" Predicted Premium for Age 68 : " + predictedData.PredictedPremium);

        }

        public static void Lab7_SavingModel()
        {
            var mlcontext = new MLContext(); 
            string filePath = "C:\\Users\\prach\\source\\repos\\Ml.Net\\SampleData\\linear_insurance_100k.csv";

            var data = mlcontext.Data.LoadFromTextFile<InsuranceData>(filePath, hasHeader: true, separatorChar: ',');// load data from csv file.

            var splitData = mlcontext.Data.TrainTestSplit(data, testFraction: 0.2);

            var trainData = splitData.TrainSet;

            var testData = splitData.TestSet;

            var pipeline = mlcontext.Transforms.Concatenate("f1", "Age")
                                               .Append(mlcontext.Regression
                                               .Trainers.Ols(labelColumnName: "Premium", featureColumnName: "f1"));
            var model = pipeline.Fit(data);

            mlcontext.Model.Save(model, data.Schema, "insuranceModel.zip");

            var pedictions = mlcontext.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);

            var predictedData = pedictions.Predict(new InsuranceData() { Age = 68 });

            Console.Write(" Predicted Premium for Age 68 : " + predictedData.PredictedPremium);

        }

        public static void Lab7_LoadingModel()
        { 
            var mlContext = new MLContext();
            
            // LOAD OLD MODEL
            DataViewSchema inputSchema;
            var loadedModel = mlContext.Model.Load("insuranceModel.zip", out inputSchema);

            // NEW TRAINING DATA (new rows)
            var newData = new List<InsuranceData>
            {
               new InsuranceData { Age = 120, Premium = 70000 },
            };

            var newDataView = mlContext.Data.LoadFromEnumerable(newData);

            // RETRAIN (INCREMENTAL FIT)
            var trainer = mlContext.Regression.Trainers.OnlineGradientDescent(labelColumnName: "Premium", featureColumnName: "f1");

            var modelChain = (TransformerChain<ITransformer>)loadedModel;//TransformerChain holds sequence of transformers (data transformations and the final predictor)

            // 1. Prepare the new data using the same transformations as the original model.
            IDataView preppedNewDataView = loadedModel.Transform(newDataView);

            // 2. Get the last transformer in the chain, which is the actual trained predictor.
            ITransformer finalPredictor = modelChain.Last();

            // 3. Cast the final predictor to the specific interface that holds the 'Model' property.
            // We assume object as the output type for safety, it varies by scenario.
            var singleFeaturePredictor = (ISingleFeaturePredictionTransformer<object>)finalPredictor;

            // 4. Finally, access the specific Model Parameters type.
            LinearRegressionModelParameters originalModelParameters = singleFeaturePredictor.Model as LinearRegressionModelParameters;

            var model2 = trainer.Fit(preppedNewDataView, originalModelParameters);

            var pe = mlContext.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(model2);

            var prediction = pe.Predict(new InsuranceData { Age = 120 });

            Console.WriteLine(prediction.PredictedPremium);

            Console.WriteLine("Model updated!");
        }

        public static void Lab8_LogisticCalssification()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromEnumerable(RegressionData.GetFruitData());

            var piprline = mlContext.Transforms.Concatenate("Feature","Weight")
                                                  .Append(mlContext.BinaryClassification
                                                  .Trainers.LbfgsLogisticRegression(labelColumnName: "IsApple", featureColumnName: "Feature"));
            var model = piprline.Fit(data);

            //var predictionEngine = mlContext.Model.CreatePredictionEngine<FruitData, FruitPrediction>(model);

            var prdictionView = mlContext.Data.CreateEnumerable<FruitData>(data, reuseRowObject: false).ToList();

            foreach(var run in prdictionView)
            {
                var predictionEngine = mlContext.Model.CreatePredictionEngine<FruitData, FruitPrediction>(model);

                var prediction = predictionEngine.Predict(new FruitData() { Weight = run.Weight, Color=run.Color });

               // Console.WriteLine($"Weight: {run.Weight} | Color: {run.Color} | IsApple: {prediction.IsApple}");
            }

           // var prediction = predictionEngine.Predict(new FruitData() { Weight = 150 });

           // Console.WriteLine("Is Apple : " + prediction.IsApple);

            Console.ReadLine();
        }

        public static void Lab9_MulticlassCalssification()
        {
            var mlContext = new MLContext();

            var data = mlContext.Data.LoadFromEnumerable(RegressionData.GetFruitData());

            var pipeline =   mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(FruitData.FruitType))//convert string labels to key types
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding("ColorEncoded", nameof(FruitData.Color)))
                            .Append(mlContext.Transforms.Concatenate("Features", "Weight", "ColorEncoded"))
                            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())//multiclass classification using SDCA maximum entropy algorithm
                            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));//convert predicted label back to original string values

            var model = pipeline.Fit(data);
            var engine = mlContext.Model.CreatePredictionEngine<FruitData, FruitPrediction>(model);

            var testData = new FruitData
            {
                Weight = 110,
                Color = "Yellow",

            };

            var result = engine.Predict(testData);

            Console.WriteLine($"Predicted Type: {result.PredictedLabel}");

        }

        public static void Lab10_SimpleCustering()
        {
            var mlContext = new MLContext();

            var data = mlContext.Data.LoadFromEnumerable(RegressionData.GetCustomerData());

            var pipeline = mlContext.Transforms.Concatenate("Features", "Age", "Spending")
                .Append(mlContext.Clustering.Trainers.KMeans(numberOfClusters: 4));//Kmeans clustering with 3 clusters . KMeans is an unsupervised learning algorithm that groups data points into clusters based on feature similarity.

            var model = pipeline.Fit(data);

            var engine = mlContext.Model.CreatePredictionEngine<CustomerData, CustomerCluster>(model);

            var prdectionView = mlContext.Data.CreateEnumerable<CustomerData>(data, reuseRowObject: false).ToList();

            foreach (var run in prdectionView)
            {
               var prediction = engine.Predict(new CustomerData { Age = run.Age, Spending = run.Spending });
               Console.WriteLine($"Age: {run.Age} | Spending: {run.Spending} | Cluster: {prediction.PredictedClusterId}");
            }

            var test = new CustomerData { Age = 35, Spending = 35000 };

            var result = engine.Predict(test);

            Console.WriteLine($"Age: 35 | Spending: 35000 |Cluster: {result.PredictedClusterId}");
        }
    }
}
