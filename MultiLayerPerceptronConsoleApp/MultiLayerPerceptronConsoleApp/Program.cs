using System;

namespace MultiLayerPerceptronConsoleApp
{
    class Program
    {

        public static double Sigmoid(double x, double b = 1)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double Derivative(double x) {
      
            return x * (1.0 - x);    
        }


        
        static void Main(string[] args)
        {

            double[,] input = new double[,] {
                {0, 0}, 
                {0, 1}, 
                {1, 0}, 
                {1, 1}
            };

            double[] output = new double[] { 0, 1, 1, 0 };

            Random r = new Random();

            double[,] hiddenWeights = {

                { r.NextDouble(), r.NextDouble(), r.NextDouble() },
                { r.NextDouble(), r.NextDouble(), r.NextDouble() },
            };

            double[] outputWeights = new double[] {

                 r.NextDouble(), r.NextDouble(), r.NextDouble()
            };

            int epochs = 0;
            int maxEpochs = 100;

            double momentum = 1.0;
            double learningRate = 0.3;

          

            while (epochs++ < maxEpochs)
            {

                for (int i = 0; i < 4; i++)
                {
                    double x1 = input[i, 0];
                    double x2 = input[i, 1];

                    double y1 = Sigmoid(x1 * hiddenWeights[0, 0] + x2 * hiddenWeights[1, 0]);
                    double y2 = Sigmoid(x1 * hiddenWeights[0, 1] + x2 * hiddenWeights[1, 1]);
                    double y3 = Sigmoid(x1 * hiddenWeights[0, 2] + x2 * hiddenWeights[1, 2]);
                    double y4 = Sigmoid(x1 * outputWeights[0] +
                                        y2 * outputWeights[1] +
                                        y3 * outputWeights[2]);


                    double error = output[i] - y4;

                    double d4 = Derivative(y4) * error;
                    double d3 = Derivative(y3) * error;
                    double d2 = Derivative(y2) * error;
                    double d1 = Derivative(y1) * error;

                    outputWeights[0] = (outputWeights[0] * momentum) + (y1 * d4 * learningRate);
                    outputWeights[1] = (outputWeights[1] * momentum) + (y2 * d4 * learningRate);
                    outputWeights[2] = (outputWeights[2] * momentum) + (y3 * d4 * learningRate);

                    hiddenWeights[0, 0] = (hiddenWeights[0, 0] * momentum) + (x1 * d1 * learningRate);
                    hiddenWeights[0, 1] = (hiddenWeights[0, 1] * momentum) + (x1 * d2 * learningRate);
                    hiddenWeights[0, 2] = (hiddenWeights[0, 2] * momentum) + (x1 * d3 * learningRate);

                    hiddenWeights[1, 0] = (hiddenWeights[1, 0] * momentum) + (x2 * d1 * learningRate);
                    hiddenWeights[1, 1] = (hiddenWeights[1, 1] * momentum) + (x2 * d2 * learningRate);
                    hiddenWeights[1, 2] = (hiddenWeights[1, 2] * momentum) + (x2 * d3 * learningRate);

                    Console.WriteLine("E({0:0.000}):\tX1({1}) XOR X2 ({2}) = Y({3:0.0})",
                        error, x1, x2, y4);

                }

                Console.WriteLine("----------------------------------------------");

            }

            Console.ReadKey();
        }
    }
}
