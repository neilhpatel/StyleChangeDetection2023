from baselineCompressor import train as baseline_compressor_train
from baselineCompressor import test as baseline_compressor_test

from baselineCNGDist import train as baseline_cngdist_train
from baselineCNGDist import test as baseline_cngdist_test

import os

class CEBaseline():

    def __init__(self, task, args):
        self.task = task
        self.dropout = args.dropout if args.dropout else .5
        self.num_iterations = args.num_iterations if args.num_iterations else 0
        self.vocabulary_size = args.vocabulary_size if args.vocabulary_size else 3000
        self.ppm_order = args.ppmOrder if args.ppmOrder else 5
        self.ngram_size = args.ngram_size if args.ngram_size else 4

    def createModel(self, baseline_type):
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

        if baseline_type == "compressor":
            baseline_compressor_train(inputTrainingData, inputLabels, f'{self.task.compressor_model_dir}/Task{self.task.task_num}.model', self.ppm_order)
        elif baseline_type == "cngdist":
            baseline_cngdist_train(inputTrainingData, inputLabels, self.task.cngdist_model_dir, self.task.task_num,
                                   self.vocabulary_size, self.ngram_size, self.num_iterations, self.dropout)
        else:
            raise AssertionError("invalid baseline_type")

    def testModel(self, baseline_type):
        testTextDataDict = self.task.paragraphs['test']
        if baseline_type == 'compressor':
            modelPath = os.path.abspath(os.getcwd()) + os.sep + self.task.compressor_model_dir
            predictedValues = baseline_compressor_test(f'{modelPath}/Task{self.task.task_num}.model', testTextDataDict, radius=0.01)
            self.task.fileReader.writeSolutionFolder(self.task, predictedValues, "compressor")
        elif baseline_type == 'cngdist':
            modelPath = os.path.abspath(os.getcwd()) + os.sep + self.task.cngdist_model_dir
            predictedValues = baseline_cngdist_test(modelPath, testTextDataDict, self.task.task_num, num_iterations=0) # dict of doc id map to dict of pair ids map to preds
            fileNumToSolutionDict = {}
            for (k, v) in predictedValues.items():
                fileNum = k
                changesList = []
                for (index, binaryValue) in v.items():
                    changesList.append(binaryValue)

                fileNumToSolutionDict[fileNum] = changesList

            self.task.fileReader.writeSolutionFolder(self.task, fileNumToSolutionDict, "cngdist")
        else:
            raise AssertionError("invalid baseline_type")

    def printcngdistStatistics(self, predictedValues):
        num_pairs_ground_truth = 0
        num_pairs_pred_vals = 0
        test_dict = self.task.data_split_dict['test']
        for key in test_dict:
            num_pairs_ground_truth += len(test_dict[key]['changes'])
            num_pairs_pred_vals += len(predictedValues[key])
            if len(predictedValues[key]) != len(test_dict[key]['changes']):
                print("unequal num preds and ground truth for key", key)
        print("num_pairs_ground_truth:", num_pairs_ground_truth)
        print("num_pairs_pred_vals:", num_pairs_pred_vals)

