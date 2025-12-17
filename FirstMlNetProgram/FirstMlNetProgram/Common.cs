using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FirstMlNetProgram
{
    public static class Common
    {
        public static string  GetDataPath()
        {
            string basePath = Directory.GetCurrentDirectory();

            string projectRoot = Directory.GetParent(basePath).Parent.Parent.FullName;

            var dataPath = Path.Combine(projectRoot, "Data");

           return dataPath;
        }
        public static double CalculateCosineSimilarity(float[] vector1, float[] vector2) 
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Vectors must be of the same length.");
            }

            double dotProduct = 0.0;
            double magnitude1 = 0.0;
            double magnitude2 = 0.0;

            for (int i = 0; i < vector1.Length; i++)
            {
                dotProduct += vector1[i] * vector2[i];
                magnitude1 += vector1[i] * vector1[i];
                magnitude2 += vector2[i] * vector2[i];
            }

            magnitude1 = Math.Sqrt(magnitude1);
            magnitude2 = Math.Sqrt(magnitude2);

            if (magnitude1 == 0.0 || magnitude2 == 0.0)
            {
                // Handle cases where one or both vectors are zero vectors
                return 0.0;
            }

            return dotProduct / (magnitude1 * magnitude2);
        }

        public static double CalculateEuclideanDistance(float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Vectors must be of the same length.");
            }

            double sumOfSquares = 0.0;
            for (int i = 0; i < vector1.Length; i++)
            {
                double difference = vector1[i] - vector2[i];
                sumOfSquares += difference * difference;
            }

            return Math.Sqrt(sumOfSquares);
        }

    }
}
