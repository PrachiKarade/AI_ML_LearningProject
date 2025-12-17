using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace FirstMlNetProgram.Cohort.Cohort_Models
{
    public class NiftyModels
    {
        public class NiftyData
        {
            [LoadColumn(1)]
            public float Nifty { get; set; }   // Historical price value
        }

        public class NiftyForecast
        {
            [ColumnName("ForecastedNifty")]
            public float[] ForecastedNifty { get; set; }
        }

        public class NiftyLagData
        {
            [LoadColumn(0)] public string Date { get; set; }
            [LoadColumn(1)] public float Nifty { get; set; }

            [LoadColumn(2)] public float NiftyLag1 { get; set; }
            [LoadColumn(3)] public float NiftyLag2 { get; set; }
            [LoadColumn(4)] public float NiftyLag3 { get; set; }
        }

        public class NiftyPrediction
        {
            [ColumnName("Score")]
            public float PredictedValue { get; set; }
        }
    }
}
