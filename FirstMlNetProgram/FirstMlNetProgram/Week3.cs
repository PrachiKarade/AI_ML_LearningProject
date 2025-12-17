using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using FirstMlNetProgram.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using static FirstMlNetProgram.Models.NLPModels;

namespace FirstMlNetProgram
{
    public class Week3
    {

        public static void Lab11_OnHotEncoding()
        {
            var mlContext = new MLContext();

            var data = new[]
            {
                new NLPModels.FruitData { Fruit = "Mango" },
                new NLPModels.FruitData { Fruit = "Apple" },
                new NLPModels.FruitData { Fruit = "Berry" },
                new NLPModels.FruitData { Fruit = "Berry" },
            };

            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "Fruit", outputColumnName: "FruitEncoded");
            var model = pipeline.Fit(dataView);

            var transformedData = model.Transform(dataView);

            var encodedData = mlContext.Data.CreateEnumerable<NLPModels.FruitFeatures>(transformedData, reuseRowObject: false).ToList();

            Console.WriteLine("One-Hot Encoded Vectors:");
            foreach (var row in encodedData)
            {
               Console.WriteLine($"[{string.Join(",", row.FruitEncoded)}]");
            }
        }

        public static void Lab12and13_BowTFIDF()
        { 
            var mlContext = new MLContext();

            var data = new[]
            {
                new NLPModels.InputText { Text = "I love programming in C#" },
                new NLPModels.InputText { Text = "C# is a great programming language" },
                new NLPModels.InputText { Text = "I enjoy learning new programming languages" },
            };
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var bowPipeline =  mlContext.Transforms.Text.TokenizeIntoWords(outputColumnName: "Tokens",inputColumnName:"Text")
                                                        .Append(mlContext.Transforms.Conversion.MapValueToKey("KeyTokens", "Tokens"))
                                                        .Append(mlContext.Transforms.Text
                                                        .ProduceNgrams(outputColumnName: "Features", inputColumnName: "KeyTokens",
                                                        ngramLength: 1, useAllLengths: false,  // Do NOT create bigrams
                                                        weighting: Microsoft.ML.Transforms.Text.NgramExtractingEstimator.WeightingCriteria.TfIdf ));

            var bowModel = bowPipeline.Fit(dataView);

            var bowTransformed = bowModel.Transform(dataView);

            var bowResults = mlContext.Data.CreateEnumerable<TextFeatures>(bowTransformed, reuseRowObject: false);

            VBuffer<ReadOnlyMemory<char>> slotNames = default; // to hold the vocabulary vBuffer is a structure used by ML.NET to efficiently handle vectors

            bowTransformed.Schema["Features"].Annotations.GetValue("SlotNames", ref slotNames); // extract the vocabulary from the "Features" column

            var vocab = slotNames.DenseValues().Select(v => v.ToString()).ToArray();// convert ReadOnlyMemory<char> to string array


            Console.WriteLine(" *** Vocabulary:*** " );

            Console.WriteLine( string.Join(", ", vocab));

            Console.WriteLine(" ------------------------------------------------------------------ ");

            foreach (var row in bowResults)
            {
                Console.WriteLine(" ******* Features:******* " );
                Console.WriteLine(string.Join(", ", row.Features));
            }

            int docIndex = 1;
            foreach (var row in bowResults) //loop through each document
            {
                Console.WriteLine(" ------------------------------------------------------------------ ");
                Console.WriteLine($"--- Document {docIndex++} ---");

                for (int i = 0; i < vocab.Length; i++) // loop through each word in the vocabulary
                {
                    if (row.Features[i] != 0) // if word is present in the document
                    {
                        Console.WriteLine($"{vocab[i]} : {row.Features[i]}"); // print word and its TF-IDF score both arrays share same index
                    }
                }
            }
        }

        public static void Lab14_Embedding()
        {
            var mlContext = new MLContext();

            var samples = new[]
            {
                new NLPModels.InputText { Text = "king" },
                new NLPModels.InputText { Text = "queen" },
                new NLPModels.InputText { Text = "camera" }
            };

            var data = mlContext.Data.LoadFromEnumerable(samples);

            var tokenizationPipeline = mlContext.Transforms.Text.TokenizeIntoWords(outputColumnName: "Tokens",inputColumnName: "Text");

            var embeddingPipeline = mlContext.Transforms
                                              .Text
                                              .ApplyWordEmbedding
                                              (
                                                outputColumnName: "Features",inputColumnName: "Tokens",
                                                modelKind: Microsoft.ML.Transforms.Text.WordEmbeddingEstimator.PretrainedModelKind.GloVe50D
                                              );

            var pipeline = tokenizationPipeline.Append(embeddingPipeline);
            var model = pipeline.Fit(data);
            var transformed = model.Transform(data);

            var results = mlContext.Data.CreateEnumerable<TextFeatures>(transformed, false).ToList();


            for (int i = 0; i < results.Count; i++)
            {
                Console.WriteLine($"\nWord: {samples[i].Text}");

                Console.WriteLine("Vector (first 10 values):");

                Console.WriteLine(string.Join(", ", results[i].Features.Take(10)) + " ...");
            }
            var resultsList = results.ToList();

            var kingVector   = resultsList[0].Features;
            var queenVector  = resultsList[1].Features;
            var cameraVector = resultsList[2].Features;

            double distanceKingQueen = Common.CalculateCosineSimilarity(kingVector, queenVector);
            double distanceKingCamera = Common.CalculateCosineSimilarity(kingVector, cameraVector);

            Console.WriteLine($"\nDistance (King vs. Queen): {distanceKingQueen:F4}");
            Console.WriteLine($"Distance (King vs. Camera): {distanceKingCamera:F4}");
        }

    }
}
