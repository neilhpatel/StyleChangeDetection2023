from fileReader import fileReader
import os

class Task(object):

    def __init__(self, task_num):
        self.task_num = task_num #1, 2, 3
        self.compressor_model_dir = f'models/baseline/compressor' #location to save the model
        self.cngdist_model_dir = f'models/baseline/cngdist' #location to save the model
        self.testSolutionDir = f'/solution/CEBaseline/Task{self.task_num}' #location to save the predicted labels for test data
        self.datasetTrainFilePath = os.getcwd() + f'/release/pan23-multi-author-analysis-dataset{self.task_num}/pan23-multi-author-analysis-dataset{self.task_num}-train'
        self.datasetValidationFilePath = os.getcwd() + f'/release/pan23-multi-author-analysis-dataset{self.task_num}/pan23-multi-author-analysis-dataset{self.task_num}-validation'
        self.paragraphs = {'train': {}, 'val': {}, 'test': {}}  # maps txt file to a list of paragraph text (dict[str, list])
        self.paragraphLength = {}  # maps txt file to an int representing the number of paragraphs (dict[str, int])
        self.JSONContents = {'train': {},'val': {}}  # maps JSON file to dictionary containing authors and changes (dict[str, {dict['authors', int], dict['changes', list]}])
        self.fileReader = fileReader()  # fileReader object that reads txt and JSON files
        self.origTrainFiles = self.fileReader.removeDSFiles(os.listdir(self.datasetTrainFilePath))
        self.origValFiles = self.fileReader.removeDSFiles(os.listdir(self.datasetValidationFilePath))
        self.negativeExamples = 0 #Negative examples occur when two consecutive paragraphs are written by the same author (value of 0)
        self.positiveExamples = 0 #Positive examples occur when two consecutive paragraphs are written by different author (value of 1)
        self.data_split_dict = {'train': {}, 'val': {}, 'test': {}} #Maps file number to corresponding JSON solution

    '''
    Populate all class variables by reading through the file directory for all JSON files
    '''
    def populateOriginalSplitJSONData(self):
        for file in self.origTrainFiles:
            fullFilePath = self.datasetTrainFilePath + '/' + file
            self.fileReader.readJSONFile(fullFilePath, self, 'train')

        for file in self.origValFiles:
            fullFilePath = self.datasetValidationFilePath + '/' + file
            self.fileReader.readJSONFile(fullFilePath, self, 'val')
    '''
    Split data into training, validation, and testing by populating all dictionaries
    '''
    def runCustomSplitCalcs(self):
        self.populateOriginalSplitJSONData()
        self.fileReader.calc_num_neg_pos_examples(self, 'train')
        self.fileReader.calc_num_neg_pos_examples(self, 'val')
        self.fileReader.split_data(self)

        for file in self.origTrainFiles:
            if file.lower().endswith('txt'):
                file_num = int(file.split('-')[1].split('.')[0])
                fullFilePath = self.datasetTrainFilePath + '/' + file
                if file_num in self.data_split_dict['train'].keys():
                    self.fileReader.readTxtFile(fullFilePath, self, 'train')
                elif file_num in self.data_split_dict['val'].keys():
                    self.fileReader.readTxtFile(fullFilePath, self, 'val')
                else:
                    raise IOError("key error:", fullFilePath)
        for file in self.origValFiles:
            if file.lower().endswith('txt'):
                file_num = file.split('-')[1].split('.')[0]
                fullFilePath = self.datasetValidationFilePath + '/' + file
                self.fileReader.readTxtFile(fullFilePath, self, 'test')

    # Print statistics below and confirm no missing files for any tasks:
    # print(len(task.data_split_dict['train']))
    # print("data split train keys:", task.data_split_dict['train'].keys())
    # print()
    # print(len(task.data_split_dict['val']))
    # print("data split dict val keys:", task.data_split_dict['val'].keys())
    # print()
    # print(len(task.data_split_dict['test']))
    # print("data split dict test keys:", task.data_split_dict['test'].keys())
    # print()
    # print()
    # print()
    # print(len(task.paragraphs['train']))
    # print("paragraphs train keys:", task.paragraphs['train'].keys())
    # print()
    # print(len(task.paragraphs['val']))
    # print("paragraphs val keys:", task.paragraphs['val'].keys())
    # print()
    # print(len(task.paragraphs['test']))
    # print("paragraphs test keys:", task.paragraphs['test'].keys())

    # for val in task.paragraphs['test']:
    #     if val not in task.data_split_dict['test']:
    #         print("missing test key:", val)
    # for val in task.paragraphs['train']:
    #     if val not in task.data_split_dict['train']:
    #         print("missing train key:", val)
    # for val in task.paragraphs['val']:
    #     if val not in task.data_split_dict['val']:
    #         print("missing val key:", val)

    # task2 = Task(2)
    # runCustomSplitCalcs(task2)

    # task3 = Task(3)
    # runCustomSplitCalcs(task3)

    # task1.populateData()
    # task1.fileReader.calc_num_neg_pos_examples(task1, 'train')
    # task1.fileReader.calc_num_neg_pos_examples(task1, 'val')
    # task1.fileReader.split_data(task1)
    # print(len(task1.data_split_dict['train']))
    # print("train keys:", task1.data_split_dict['train'].keys())
    # print()
    # print(len(task1.data_split_dict['val']))
    # print("val keys:", task1.data_split_dict['val'].keys())
    # print()
    # print(len(task1.data_split_dict['test']))
    # print("test keys:", task1.data_split_dict['test'].keys())
    # print()