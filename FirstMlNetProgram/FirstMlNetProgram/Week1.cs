using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FirstMlNetProgram.Models;
using FirstMlNetProgram.Data;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace FirstMlNetProgram
{
    public class Week1
    {
        public static void Lab1_Simples_MLCode_Using_SinglePrediction()
        {  

            var mlcontext = new MLContext(); //create ML context :mlcontext is like a starting point for all ML.NET operations
            
            var contextData = mlcontext.Data.LoadFromEnumerable(RegressionData.GetInsuranceData());//load data into ML.NET environment

            //step 3: define data preparation and model training pipeline .pipeline is a sequence of data transformations and model training operations
            var pipeline = mlcontext.Transforms.Concatenate("f1", "Age")// feature engineering step:combine input features into a single feature vector
                                               .Append(mlcontext.Regression //regression task:predict continuous value
                                               .Trainers.Ols(labelColumnName: "Premium", featureColumnName: "f1"));//use ordinary least squares(OLS) regression algorithm
            //step 4: train the model
            var model = pipeline.Fit(contextData);//fit the pipeline to the data to create a trained model

            //step 5: make predictions
            var predictionFunction = mlcontext.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(model); //create a prediction engine to make predictions on new data

            //step 6: test the model with a sample input
            var prediction = predictionFunction.Predict(new InsuranceData() { Age = 25 });

            Console.WriteLine(prediction.PredictedPremium);
            Console.Read();
        }

        public static void Lab2_Simples_MLCode_Using_TestData()
        {

            var mlcontext = new MLContext(); //create ML context :mlcontext is like a starting point for all ML.NET operations

            var contextData = mlcontext.Data.LoadFromEnumerable(RegressionData.GetInsuranceData());//load data into ML.NET environment
            var testData = mlcontext.Data.LoadFromEnumerable(RegressionData.GetTestData());//load test data into ML.NET environment

            //step 3: define data preparation and model training pipeline .pipeline is a sequence of data transformations and model training operations
            var pipeline = mlcontext.Transforms.Concatenate("f1", "Age")// feature engineering step:combine input features into a single feature vector
                                               .Append(mlcontext.Regression //regression task:predict continuous value
                                               .Trainers.Ols(labelColumnName: "Premium", featureColumnName: "f1"));//use ordinary least squares(OLS) regression algorithm
            //step 4: train the model
            var model = pipeline.Fit(contextData);//fit the pipeline to the data to create a trained model
            var prediction = model.Transform(testData); // Transform returns IDataView, which is a tabular representation of data in ML.NET

            var predictionEnumerable = mlcontext.Data.CreateEnumerable<InsurancePrediction>(prediction, reuseRowObject: false).ToList();

            //step 5: take  prediction lists

            foreach (var predectedData in predictionEnumerable)
            {
                Console.WriteLine(predectedData.PredictedPremium);
            }
            Console.ReadLine();
        }

        public static void Lab3_Simples_MLCode_Checking_RSandRMSE()
        {
            var mlContext = new MLContext(); //create ML context :mlcontext is like a starting point for all ML.NET operations

            var data = mlContext.Data.LoadFromEnumerable(RegressionData.GetInsuranceData());// load data 

            var testData = mlContext.Data.LoadFromEnumerable(RegressionData.GetTestData()); //load test data 

            var pipeline = mlContext.Transforms.Concatenate("feature", "Age")
                                    .Append(mlContext.Regression.Trainers.Ols(labelColumnName: "Premium", featureColumnName: "feature"));

            var model = pipeline.Fit(data);

            IDataView predictions = model.Transform(testData); // model.Transform returns IDataView, which is a tabular representation of data in ML.NET

            var metrics = mlContext.Regression.Evaluate(predictions,labelColumnName: "Premium",scoreColumnName: "Score");

            Console.WriteLine("R Squared :" + metrics.RSquared);
            Console.WriteLine("RMSE = RootMeanSquaredError :" + metrics.RootMeanSquaredError);// RMSE 
            Console.ReadLine();
        }

        public static void Lab5_SimplestMLAutoMl()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(RegressionData.GetInsuranceData());
            var testData = mlcontext.Data.LoadFromEnumerable(RegressionData.GetTestData());

            var experimentSetting = new RegressionExperimentSettings { MaxExperimentTimeInSeconds = 30 };

            var experiment = mlcontext.Auto().CreateRegressionExperiment(experimentSetting);

            var result = experiment.Execute(data,labelColumnName:"Premium");

            foreach(var run in result.RunDetails)
            {
                Console.WriteLine(" -----------------------------------------------------");
                Console.WriteLine(" Trainer Name: " + run.TrainerName);
                Console.WriteLine(" ----------- Metrics ------------------------");
                Console.WriteLine(" RSquared: " + run.ValidationMetrics.RSquared);
                Console.WriteLine(" RootMeanSquaredError: " + run.ValidationMetrics.RootMeanSquaredError);                
            }
            Console.WriteLine(" -----------------------------------------------------");
            Console.WriteLine(" Best Model: " + result.BestRun.TrainerName);
            Console.ReadLine();

        }
    }
}
