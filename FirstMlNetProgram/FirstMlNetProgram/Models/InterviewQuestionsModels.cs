using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace FirstMlNetProgram.Models
{
    public class InterviewQuestionsModels
    {
        public class QAItem
        {
            public string Question { get; set; }
            public string Answer { get; set; }
            public float[] Embedding { get; set; }
        }

        public class InputText
        {
           public string Text { get; set; }
        }

        public class TextEmbedding
        {
            [VectorType(384)] // embedding dimension
            public float[] Features { get; set; }
        }

        public class BertInput
        {
            [VectorType(128)]
            public long[] input_ids { get; set; }

            [VectorType(128)]
            public long[] attention_mask { get; set; }
        }

        public class BertOutput
        {
            [VectorType(384)]
            public float[] sentence_embedding { get; set; }
        }

    }
}
