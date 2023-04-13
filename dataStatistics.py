import math
import os
import re
import json


'''
Initial file used to read files and print statistics about the data. Ignore and use Task1.py, Task2.py, Task3.py, 
and fileReader.py.
'''

class dataStatistics:

    def __init__(self):
        self.paragraphLength = {} #maps file paths to number of paragraphs in the file (dict[str, int])
        self.paragraphs = {} #maps file to a list of paragraph text (dict[str, list])
        self.words = {} #maps file to a list of words in file (dict[str, list])
        self.JSONContents = {} #maps JSON file to dictionary containing authors and changes (dict[str, {dict['authors', int], dict['changes', list]}])


    def readTxtFile(self, txt):
        words = []
        with open(txt) as file:
            paragraphsRaw = file.read()

        with open(txt, 'r') as file:
            for line in file:
                for word in line.split():
                    words.append(word)

        file.close()
        paragraphs = paragraphsRaw.split("\n")
        self.paragraphLength[txt] = len(paragraphs)
        self.paragraphs[txt] = paragraphs
        self.words[txt] = words
        #print("Number of paragraphs in " + txt + " is: " + str(len(paragraphs)))

    def readJSON(self, JSONPath):
        file = open(JSONPath)
        data = json.load(file)
        self.JSONContents[JSONPath] = data

    def removeDSFiles(self, fileList):
        if '.DS_Store' in fileList:
            fileList.remove('.DS_Store')
        return fileList

    def printStatistics(self):
        paragraphLengths = self.paragraphLength.values()
        print("Minimum Paragraph Length is: " + str(min(paragraphLengths)))
        print("Maximum Paragraph Length is: " + str(max(paragraphLengths)))
        print(len([i for i in paragraphLengths if i == 2]))
        allWords = self.words.values()
        maxWords = 0
        minWords = math.inf
        for words in allWords:
            if len(words) > maxWords:
                maxWords = len(words)

            if len(words) < minWords:
                minWords = len(words)

        print("Minimum Word Length is: " + str(minWords))
        print("Maximum Word Length is: " + str(maxWords))

        maxAuthors = 0
        minAuthors = math.inf

        for key in self.JSONContents:
            authorCount = self.JSONContents[key]['authors']

            if authorCount > maxAuthors:
                maxAuthors = authorCount

            if authorCount < minAuthors:
                minAuthors = authorCount

        print("Minimum Author Count is: " + str(minAuthors))
        print("Maximum Author Count is: " + str(maxAuthors))



if __name__ == "__main__":
    dataClass = dataStatistics()

    dirPath = os.getcwd() + '/release'
    folders = os.listdir(dirPath)
    folders = dataClass.removeDSFiles(folders)
    for folder in folders:
        #folder is of the form pan23-multi-author-analysis-dataset
        subFolders = os.listdir(dirPath + '/' + str(folder))
        subFolders = dataClass.removeDSFiles(subFolders)
        for dataset in subFolders:
            #dataset is of the form pan23-multi-author-analysis-dataset#-train/validation
            datasetPath = dirPath + '/' + str(folder) + '/' + str(dataset)
            files = (os.listdir(datasetPath))
            #Code below sorts the directory if you want to read through files in order
            #try:
            #    sortedDir = sorted(files, key=lambda s: int(re.search(r'\d+', s).group()))
            #except AttributeError:
            #    sortedDir = files
            for file in files:
            #file is of the form problem-#.txt or truth-problem-#.json
                filePath = datasetPath + '/' + str(file)
                if filePath.lower().endswith('.txt'):
                    dataClass.readTxtFile(filePath)
                elif filePath.lower().endswith('.json'):
                    dataClass.readJSON(filePath)

    dataClass.printStatistics()


    '''
    Folder directory shows 4200 training examples, 900 validation examples for each dataset/task:
    pan23-multi-author-analysis-dataset1-train has 4200 txt & JSON files
    pan23-multi-author-analysis-dataset1-validation has 900 txt & JSON files
    pan23-multi-author-analysis-dataset2-train has 4200 txt & JSON files
    pan23-multi-author-analysis-dataset2-validation has 900 txt & JSON files
    pan23-multi-author-analysis-dataset3-train has 4200 txt & JSON files
    pan23-multi-author-analysis-dataset3-validation has 900 txt & JSON files
    Total of 15300 documents which can be confirmed by running the code and checking the length of any of the dictionary objects
    '''
