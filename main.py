from Task import Task
from CEBaseline import CEBaseline
from evaluator import evaluatorMainWrapper
from BertModel import BertModel

if __name__ == "__main__":

    for i in range(1, 2):
        print(f'Initializing Task {i}')
        task = Task(i)
        task.runCustomSplitCalcs()
        print("after custom split calcs")
        taskBertModel = BertModel(task)
        print("after taskBertModel")
        taskBertModel.createModel()

        # taskCEBaseline = CEBaseline(task)

        #Alternate between cngdist and compressor for code below
        #taskCEBaseline.createModel("cngdist")
        #taskCEBaseline.testModel("cngdist")
        #evaluatorMainWrapper(task.testSolutioncngdistDir, task.datasetValidationFilePath, "results", "cngdist", i)

        #taskCEBaseline.createModel("compressor")
        #taskCEBaseline.testModel("compressor")
        #evaluatorMainWrapper(task.testSolutionCompressorDir, task.datasetValidationFilePath, "results", "compressor", i)
