from baselineCompressor import train, test

import os

class CEBaseline():

    def __init__(self, task):
        self.task = task

    def createModel(self):
        inputTrainingData = []
        inputLabels = []
        trainingTextDataDict = self.task.paragraphs['train']
        for k in trainingTextDataDict.keys():
            JSONFileName = 'truth-problem-' + str(k) + ".json"
            fileDict = self.task.JSONContents['train'][JSONFileName]
            groundTruth = fileDict['changes']
            paragraphs = self.task.paragraphs['train'][k]
            inputTrainingData.append(paragraphs)
            inputLabels.append(groundTruth)

        train(inputTrainingData, inputLabels, self.task.model_dir, ppm_order=5)

    def testModel(self):
        testInputData = []
        testTextDataDict = self.task.paragraphs['test']
        for k in testTextDataDict.keys():
            paragraphs = testTextDataDict[k]
            testInputData.append(paragraphs)

        modelPath = os.path.abspath(os.getcwd()) + os.sep + self.task.model_dir
        predictedValues = test(modelPath, testInputData, radius=0.01)
