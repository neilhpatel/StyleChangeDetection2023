import argparse

from Task import Task
from CEBaseline import CEBaseline
from evaluator import evaluatorMainWrapper


def main():
    parser = argparse.ArgumentParser(
        description='PAN23 Style Change Detection Task')
    parser.add_argument("-ppm", "--ppmOrder", help="Character context window size",
                        required=False)
    parser.add_argument(
        "-vocab_size", "--vocabulary_size", type=int, help="maximum features considered for CNG Dist model ordered by term frequency across the corpus",
        required=False)
    parser.add_argument(
        "-ngram", "--ngram_size", type=int, help="n-grams extracted from text data", required=False)
    parser.add_argument(
        "-iter", "--num_iterations", type=int, help="number of times to randomly sample from n-gram model", required=False)
    parser.add_argument(
        "-dropout", "--dropout", type=float, help="proportion of features used when calculating similarity", required=False)
    args = parser.parse_args()

    for i in range(1, 4):
        print(f'Initializing Task {i}')
        task = Task(i)
        task.runCustomSplitCalcs()
        taskCEBaseline = CEBaseline(task, args)

        #Alternate between cngdist and compressor for code below
        taskCEBaseline.createModel("cngdist")
        taskCEBaseline.testModel("cngdist")
        evaluatorMainWrapper(task.testSolutioncngdistDir, task.datasetValidationFilePath, "results", "cngdist", i)

        taskCEBaseline.createModel("compressor")
        taskCEBaseline.testModel("compressor")
        evaluatorMainWrapper(task.testSolutionCompressorDir, task.datasetValidationFilePath, "results", "compressor", i)

if __name__ == "__main__":
    main()

