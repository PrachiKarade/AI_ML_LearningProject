using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace FirstMlNetProgram.Models
{
    #region Insurance data
    public class InsuranceData
    {
        [LoadColumn(0)]
        public float Age { get; set; }

        [LoadColumn(1)]
        public float Premium { get; set; }
        public InsuranceData() { }
    }
    public class InsurancePrediction
    {
        [ColumnName("Score")]
        public float PredictedPremium { get; set; }
    }

    #endregion

    #region Fruit data

    public class FruitData
    {
        public float Weight { get; set; }
        public string Color { get; set; }
        public bool IsApple { get; set; }
        public string FruitType {  get; set; }
    }

    public class FruitPrediction
    {
        public string PredictedLabel { get; set; }
       // public bool IsApple { get; set; }
    }
    #endregion

    #region Customer data

    public class CustomerData
    {
        public float Age { get; set; }
        public float Spending { get; set; }
    }

    public class CustomerCluster
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId { get; set; }
    }

    #endregion
}
