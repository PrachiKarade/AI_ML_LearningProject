using System;
using System.Data;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML;
using static FirstMlNetProgram.Models.InterviewQuestionsModels;
using InputText = FirstMlNetProgram.Models.InterviewQuestionsModels.InputText;
using OpenAI;
using OpenAI.Chat;
using Google.Protobuf;
using AllMiniLmL6V2Sharp.Tokenizer;
using AllMiniLmL6V2Sharp;
using OpenAI.Embeddings;
using FirstMlNetProgram.Data;
using static FirstMlNetProgram.Models.NLPModels;



namespace FirstMlNetProgram
{
    public static class Week4
    {
        //The vocab.txt is used by the tokenizer to convert text into tokens (numbers),
        //and the ONNX model all-MiniLM-L6-v2.onnx is used by the embedder to
        //convert those tokens into vector embeddings that capture meaning.
        // https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx/tree/main       
        // from huggingface

        public static void Lab15_BertEncoding()
        {
            var dataPath = Common.GetDataPath();

            var tokenizer = new BertTokenizer(dataPath + @"\\vocab.txt"); // Load tokenizer vocab file

            var embedder = new AllMiniLmL6V2Embedder(dataPath + @"\\all-MiniLM-L6-v2.onnx",tokenizer );

            // Our small question–answer list
            var qaDatabase = new List<QAItem>
            {
                new QAItem { Question = "What is Dependency Injection in C#?", Answer = "DI allows decoupling dependencies by injecting them." },
                new QAItem { Question = "What is async/await in C#?", Answer = "async/await enables asynchronous programming." },
                new QAItem { Question = "What are SOLID principles?", Answer = "SOLID are 5 design principles for maintainable code." },
                new QAItem { Question = "Liskov in SOLID?", Answer = "SOLID are 5 design principles for maintainable code." }
            };

            // Make embeddings for each stored question
            foreach (var item in qaDatabase)
            { 
              item.Embedding = embedder.GenerateEmbedding(item.Question).ToArray();// Tokenization happens inside GenerateEmbedding()
            }

            // Ask user for a new question
            Console.WriteLine("Ask a C# interview question:");
            string userQuestion = Console.ReadLine() ?? "";

            // Make embedding for the user's question
            
            var userEmbedding = embedder.GenerateEmbedding(userQuestion).ToArray();// Tokenization also happens on user question

            // Find the stored question that is most similar to the user's question
            var bestMatch = qaDatabase
                            .Select(x => new { QA = x, Similarity = Common.CalculateCosineSimilarity(userEmbedding, x.Embedding) })
                            .OrderByDescending(x => x.Similarity)
                            .First()
                            .QA;

            // Show the best matching question + answer
            Console.WriteLine($"\nClosest stored question: {bestMatch.Question}");
            Console.WriteLine($"Answer: {bestMatch.Answer}");
        }

        public static async Task Lab17_ChatGPTOnline()
        {
            try
            {

                var key = Environment.GetEnvironmentVariable("OpenAiKey");
                //var client = new OpenAIClient(key);
                var chat = new ChatClient(model: "gpt-4o-mini", key);
                var messages = new List<ChatMessage>
                {
                    //new SystemChatMessage("Take c# interview.Only ask ASP.NET core question if he does not answr that ask him basic OOP. Do not repeat question once asked. Ask one question at a time. Do not answer yourself.")
                     new SystemChatMessage("Take ReactJs interview Questions. Ask Hooks and Redux as well.")
                };
                while (true)
                {
                    var completion = await chat.CompleteChatAsync(messages);

                    string questionfromchatgpt = completion.Value.Content.Last().Text;

                    messages.Add(new AssistantChatMessage(questionfromchatgpt));

                    Console.WriteLine($"{questionfromchatgpt}");

                    var userResponse = "";

                   userResponse = Console.ReadLine(); // answr
                    messages.Add(new UserChatMessage(userResponse)); // chat gpt
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }

        public static async Task Lab18_RAGChatGptOnline()
        {
            try
            { 
                var key = Environment.GetEnvironmentVariable("OpenAiKey");
                var chat = new ChatClient(model: "gpt-4o-mini", key);

                var embeddengClient = new EmbeddingClient(model: "text-embedding-3-small", apiKey: key);

                List<RagLookUp>  ragStore = NLPData.GetNLPData();

                foreach(var item in ragStore)
                {                  
                  var embedding = await embeddengClient.GenerateEmbeddingAsync(item.Description);
                  
                  item.DescriptionEmbedding = embedding.Value.ToFloats().ToArray();
                }  
            
                Console.WriteLine(" Who you are? Enter your experience ");

                string userExperience = Console.ReadLine() ?? "";
                var userEmbedding = await embeddengClient.GenerateEmbeddingAsync(userExperience);
                var userEmbeddingVector = userEmbedding.Value.ToFloats().ToArray();

                var bestMatch = ragStore
                                .Select(x => new { Rag = x, Similarity = Common.CalculateCosineSimilarity(userEmbeddingVector, x.DescriptionEmbedding) })
                                .OrderByDescending(x => x.Similarity)
                                .First()
                                .Rag;

                var messages = new List<ChatMessage>();

                messages.Add(new SystemChatMessage(bestMatch.QuestionAsked + " ask 10 questions"));

                var completion = await chat.CompleteChatAsync(messages);

                string questionfromchatgpt = completion.Value.Content.Last().Text;

                messages.Add(new AssistantChatMessage(questionfromchatgpt));

                Console.WriteLine($"{questionfromchatgpt}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }
}
