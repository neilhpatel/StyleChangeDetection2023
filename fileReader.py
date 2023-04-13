import json

class fileReader:

    def __init__(self):
        pass

    def readJSON(self, JSONPath, task):
        if not JSONPath.lower().endswith('.json'):
            raise IOError("Attempting to read invalid JSON file: " + str(JSONPath))
        file = open(JSONPath)
        data = json.load(file)
        filePathSplit = JSONPath.split("/")
        task.JSONContents[filePathSplit[-1]] = data

    def removeDSFiles(self, fileList):
        if '.DS_Store' in fileList:
            fileList.remove('.DS_Store')
        return fileList

    def readTxtFile(self, txtFilePath, task):
        if not txtFilePath.lower().endswith('.txt'):
            raise IOError("Attempting to read invalid txt file: " + str(txtFilePath))
        words = []
        with open(txtFilePath) as file:
            paragraphsRaw = file.read()

        with open(txtFilePath, 'r') as file:
            for line in file:
                for word in line.split():
                    words.append(word)

        file.close()
        paragraphs = paragraphsRaw.split("\n")
        filePathSplit = txtFilePath.split("/")
        task.paragraphs[filePathSplit[-1]] = paragraphs
        task.paragraphLength[filePathSplit[-1]] = len(paragraphs)

    def readFile(self, filePath, task):
        if filePath.lower().endswith('.json'):
            self.readJSON(filePath, task)
        elif filePath.lower().endswith('txt'):
            self.readTxtFile(filePath, task)
        else:
            raise IOError("Unknown file format attempting to be read: " + str(filePath))