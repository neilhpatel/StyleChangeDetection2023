from fileReader import fileReader
import os

class Task2(object):

    def __init__(self):
        self.dataset2TrainFilePath = os.getcwd() + '/release/pan23-multi-author-analysis-dataset2/pan23-multi-author-analysis-dataset2-train'
        self.dataset2ValidationFilePath = os.getcwd() + '/release/pan23-multi-author-analysis-dataset2/pan23-multi-author-analysis-dataset2-validation'
        self.paragraphs = {} #maps txt file to a list of paragraph text (dict[str, list]).
        self.paragraphLength = {} #maps txt file to an int representing the number of paragraphs (dict[str, int]).
        self.JSONContents = {} #maps JSON file to dictionary containing authors and changes (dict[str, {dict['authors', int], dict['changes', list]}])
        self.fileReader = fileReader() #fileReader object that reads txt and JSON files

    '''
    Populate all class variables by reading though the file directory for all Task3 files
    '''
    def populateData(self):
        trainFiles = os.listdir(self.dataset2TrainFilePath)
        trainFiles = self.fileReader.removeDSFiles(trainFiles)
        for file in trainFiles:
            fullFilePath = self.dataset2TrainFilePath + '/' + file
            self.fileReader.readFile(fullFilePath, self)


        validationFiles = os.listdir(self.dataset2ValidationFilePath)
        validationFiles = self.fileReader.removeDSFiles(validationFiles)
        for file in validationFiles:
            fullFilePath = self.dataset2ValidationFilePath + '/' + file
            self.fileReader.readFile(fullFilePath, self)
