import json
import numpy as np
import os

class fileReader:

    np.random.seed(42)

    def __init__(self):
        pass
    '''
    Reads JSON file and adds data to the task's JSON dictionary
    '''
    def readJSON(self, JSONPath, task, orig_split_type):
        if not JSONPath.lower().endswith('.json'):
            raise IOError("Attempting to read invalid JSON file: " + str(JSONPath))
        file = open(JSONPath)
        data = json.load(file)
        filePathSplit = JSONPath.split("/")
        task.JSONContents[orig_split_type][filePathSplit[-1]] = data
    '''
    Remove DS files from directory (Mac) to avoid file reading issues
    '''
    def removeDSFiles(self, fileList):
        if '.DS_Store' in fileList:
            fileList.remove('.DS_Store')
        return fileList
    '''
    Reads txt files and splits by paragraphs
    '''
    def readTxtFile(self, txtFilePath, task, custom_split_type):
        if txtFilePath.lower().endswith('.json'):
            pass
        elif not txtFilePath.lower().endswith('.txt'):
            raise IOError("Attempting to read invalid txt file: " + str(txtFilePath))

        with open(txtFilePath) as file:
            paragraphsRaw = file.read()

        #Find number of words in each file with code below:
        #words = []
        #with open(txtFilePath, 'r') as file:
        #    for line in file:
        #        for word in line.split():
        #            words.append(word)
        file.close()
        paragraphs = paragraphsRaw.split("\n")
        cleaned = []
        '''
        Files problem-3616.txt and problem-1559.txt in dataset1 train and problem-25.txt in dataset1 validation
        have spacing issue so fix with code below.
        '''
        for paragraph in paragraphs:
            if (paragraph[0] == ' '):
                #print(f"File {txtFilePath} has spacing issue")
                if (len(cleaned) > 0):
                    cleaned[-1] += paragraph #if paragraph begins with space, append to current last paragraph, not new paragraph
                else:
                    cleaned.append(paragraph)
            else:
                cleaned.append(paragraph)
        filePathSplit = txtFilePath.split("/")
        file_num = int(filePathSplit[-1].split('-')[1].split('.')[0])

        task.paragraphs[custom_split_type][file_num] = cleaned
        # task.paragraphLength[filePathSplit[-1]] = len(paragraphs)

    def readJSONFile(self, filePath, task, orig_split_type):
        if filePath.lower().endswith('txt'):
            #ignore txt files
            pass
        elif filePath.lower().endswith('.json'):
            self.readJSON(filePath, task, orig_split_type)
        else:
            raise IOError("Unknown file format attempting to be read: " + str(filePath))
    
    def calc_num_neg_pos_examples(self, task, orig_split_type):
        json_contents = task.JSONContents[orig_split_type]
        for key in json_contents:
            list_changes = json_contents[key]['changes']
            for val in list_changes:
                if val == 1:
                    task.positiveExamples += 1
                else:
                    task.negativeExamples += 1
    '''
    Split training folder into training and validation data. Keep validationn folder as testing data
    '''
    def split_data(self, task):
        perm = np.random.permutation(len(task.JSONContents['train']))
        train_split = perm[:3570] #3570
        val_split = perm[3570:] #630
        # task.data_split_dict['test'] = task.JSONContents['val']
        for val in train_split:
            file_num = val+1
            task.data_split_dict['train'][file_num] = task.JSONContents['train'][f'truth-problem-{file_num}.json']
            # task.data_split_dict['train'][f'truth-problem-{file_num}.json'] = task.JSONContents['train'][f'truth-problem-{file_num}.json']
            # task.txt_truth_dict['train'][f'problem-{file_num}.txt'] = f'truth-problem-{file_num}.json'
        for val in val_split:
            file_num = val+1
            task.data_split_dict['val'][file_num] = task.JSONContents['train'][f'truth-problem-{file_num}.json']
            # task.data_split_dict['val'][f'truth-problem-{file_num}.json'] = task.JSONContents['train'][f'truth-problem-{file_num}.json']
            # task.txt_truth_dict['val'][f'problem-{file_num}.txt'] = f'truth-problem-{file_num}.json'
        for json_filename_key in task.JSONContents['val']:
            file_num = int(json_filename_key.split('-')[2].split('.')[0])
            task.data_split_dict['test'][file_num] = task.JSONContents['val'][f'truth-problem-{file_num}.json']
            # task.txt_truth_dict['test'][f'problem-{file_num}.txt'] = f'truth-problem-{file_num}.json'




    def writeSolutionFolder(self, task, predictedValues):
        if type(predictedValues) != dict:
            raise AssertionError(f'Cannot write solution folder with given input of type {type(predictedValues)}')
        #'solution-problem-*.json'
        outputDir = os.path.abspath(os.getcwd() + os.sep + task.testSolutionDir)
        for k, v in predictedValues.items():
            fileName = f'solution-problem-{k}.json'
            filePath = outputDir + os.sep + fileName
            v = [int(val) for val in v] #convert Type int64 to int so value is JSON serializable
            key = 'changes'
            dictionaryToWrite = {key : v}
            with open(filePath, "w") as f:
                json.dump(dictionaryToWrite, f)