using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace FirstMlNetProgram.Models
{
    public class NLPModels
    {
        #region Model class for One-Hot Encoding
        public class FruitData 
        {
          public string Fruit { get; set; }
        }
        public class FruitFeatures
        {
            // [VectorType]
            public float[] FruitEncoded { get; set; }
        }
        #endregion

        #region Model class for Bag of Words and TF-IDF

        public class InputText
        {
            [LoadColumn(0)]
            public string Text { get; set; }
        }

        public class TextFeatures
        {
            [ColumnName("Features")]
            [VectorType]
            public float[] Features { get; set; }
        }
        #endregion

        public class RagLookUp
        {
            public string Description { get; set; }
            public string QuestionAsked { get; set; }
            public float[] DescriptionEmbedding { get; set; }
        }
    }
}
