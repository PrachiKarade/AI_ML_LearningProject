using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static FirstMlNetProgram.Models.NLPModels;

namespace FirstMlNetProgram.Data
{
    public  class NLPData
    {
        public static List<RagLookUp> GetNLPData()
        {
            return new List<RagLookUp>
            {
                new RagLookUp 
                { 
                  Description = "one year experience with .net developer.",
                  QuestionAsked = "C# Basic and OOPs concept." 
                },
                new RagLookUp
                { 
                  Description = "Sinior developer with more than five year experience in .net developer.",
                  QuestionAsked = "Advance asp.net , web api core , SQL Server and cloud."
                },
                new RagLookUp
                {
                  Description = "Architect level 10+ experience",
                  QuestionAsked = "Architexture level . performance tuning, Azure architecture, DDD, Distributed architecture."
                },

            };
        }
    }
}
