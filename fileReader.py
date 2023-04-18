import json
import numpy as np

class fileReader:

    np.random.seed(42)

    def __init__(self):
        pass

    def readJSON(self, JSONPath, task, orig_split_type):
        if not JSONPath.lower().endswith('.json'):
            raise IOError("Attempting to read invalid JSON file: " + str(JSONPath))
        file = open(JSONPath)
        data = json.load(file)
        filePathSplit = JSONPath.split("/")
        task.JSONContents[orig_split_type][filePathSplit[-1]] = data

    def removeDSFiles(self, fileList):
        if '.DS_Store' in fileList:
            fileList.remove('.DS_Store')
        return fileList

    def readTxtFile(self, txtFilePath, task, custom_split_type):
        if txtFilePath.lower().endswith('.json'):
            pass
        elif not txtFilePath.lower().endswith('.txt'):
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
        file_num = int(filePathSplit[-1].split('-')[1].split('.')[0])

        task.paragraphs[custom_split_type][file_num] = paragraphs
        # task.paragraphLength[filePathSplit[-1]] = len(paragraphs)

    def readJSONFile(self, filePath, task, orig_split_type):
        if filePath.lower().endswith('.json'):
            self.readJSON(filePath, task, orig_split_type)
        elif filePath.lower().endswith('txt'):
            # self.readTxtFile(filePath, task)
            pass
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




    