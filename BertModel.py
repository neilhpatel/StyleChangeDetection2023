import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import re
import os
from evaluator import compute_score_multiple_predictions

# first, install the hugging face transformer package in your colab
# pip install transformers
from transformers import get_linear_schedule_with_warmup
from tokenizers.processors import BertProcessing

# Do not change this line, as it sets the model the model that Hugging Face will load
# If you are interested in what other models are available, you can find the list of model names here:
# https://huggingface.co/transformers/pretrained_models.html
bert_model_name = "distilbert-base-uncased"
##YOUR CODE HERE##
from transformers import DistilBertModel, DistilBertTokenizer

from joblib import dump, load


# paragraphs = ["She doesnt “savage” anyone, no one is “fuming” (except readers who think twitter is the real world), and the “bigotry” is certainly a problematic word-mainly because who or what is being bigoted?",
#               "I left Ohio in 1998. Never looked back. I've lived and vacationed in Michigan for 25 years. It's great. I wouldn't go back to that backward state down south if you paid me. Although their roads are really nice. And I do miss Tony Packo's.",
#               "There are definitely crazies for sure, but I'd describe the general yooper ideology as Libertarian in an innocent sense, as opposed to more often meaning republican but not able to admit it.",
#               "It's so nice to see this. Very refreshing to finally see an elected official calling Republicans out for their bigotry and nonsense instead of wringing their hands and trying, yet again, to be best buddies. More of this, please.",
#               "“obstinate or unreasonable attachment to a belief, opinion, or faction, in particular prejudice against a person or people on the basis of their membership of a particular group”.",
#               "Yeah. Houghton and Marquette are healthy sustainable communities. The rest of the up has been a bit brain drained for many reasons."]
# labels = [1, 1, 1, 1, 1]

tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)


class BertModel(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        if torch.cuda.is_available():
          self.device = torch.device("cuda")
        else:
          self.device = torch.device("cpu")
        print("Using device:", self.device)
        

    def createModel(self):
        print("createModel")
        inputTrainingData = []
        inputLabels = []
        trainingTextDataDict = self.task.paragraphs["train"]
        for k in trainingTextDataDict.keys():
            JSONFileName = "truth-problem-" + str(k) + ".json"
            fileDict = self.task.JSONContents["train"][JSONFileName]
            groundTruth = fileDict["changes"]
            paragraphs = self.task.paragraphs["train"][k]
            inputTrainingData.append(paragraphs)
            inputLabels.append(groundTruth)

        print("after for loop")
        self.NUM_EPOCHS = 1
        print("after num epochs")
        self.bert_model = DistilBertModel.from_pretrained(bert_model_name, num_labels=2)
        print("after x")
        self.classifier = nn.Linear(2 * self.bert_model.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        print("before optimizer")
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        print("after optimizer")

        self.train(
            inputTrainingData,
            inputLabels,
            f"/content/Task{self.task.task_num}.model",
        )

    def train(self, inputTrainingData, inputLabels, modelPath):
        if len(inputTrainingData) != len(inputLabels):
            raise AssertionError("Input data and ground truth dimension mismatch")
        numDataPoints = len(inputTrainingData)
        print("numDataPoints: ", numDataPoints)
        epoch_loss = 0
        batch_size = 256
        loss = torch.zeros(1, requires_grad=True)
        for batch_start in range(0, numDataPoints, batch_size):
            print("current_batch_point: ", batch_start)
            batch_end = batch_start + batch_size
            if batch_end > numDataPoints:
                batch_end = numDataPoints
            source = inputTrainingData[batch_start:batch_end]
            target = inputLabels[batch_start:batch_end]
            self.optimizer.zero_grad()
            output = self.forward(source)
            flat_output = [item for sublist in output for item in sublist]
            flat_target = [item for sublist in target for item in sublist]
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(torch.FloatTensor(flat_output).to(self.device), torch.FloatTensor(flat_target).to(self.device))
            loss = torch.autograd.Variable(loss, requires_grad = True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            self.optimizer.step()
            print("loss: ", loss.item())
            epoch_loss += loss.item()

        path = os.path.abspath(os.getcwd())
        dump(self, modelPath)

    def forward(self, inputTrainingData):

        logits = []

        for paragraphs in inputTrainingData:
            paragraphs_score = []
            for paragraph in paragraphs:
                sentences = re.split(
                    "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", paragraph
                )
                paragraph_score = torch.zeros(self.bert_model.config.hidden_size)
                for sentence in sentences:
                    # print(len(re.split(' |.', sentence)))
                    # print(sentence)
                    if len(re.split(' |.', sentence)) > 512:
                        # print("too long sentence")
                        result = []
                        randomInd = np.random.choice(len(sentence), 505)
                        sortedInd = sorted(randomInd)
                        for ind in sortedInd:
                            result.append(sentence[ind])

                        sentence = ' '.join(result)

                    #     sentence = ' '.join(result)
                    # print(tokenizer(sentence))
                    # print(tokenizer.encode(sentence))
                    # print(tokenizer.encode(sentence, return_tensors='pt'))
                    bert_output = self.bert_model(
                        tokenizer.encode(sentence, return_tensors="pt")
                    )
                    last_hidden_state = bert_output.last_hidden_state
                    sentence_score = torch.sum(last_hidden_state[0], axis=0)
                    paragraph_score = torch.add(paragraph_score, sentence_score)
                paragraph_score = paragraph_score / len(sentences)
                paragraphs_score.append(paragraph_score)

            file_logits = []
            for i in range(len(paragraphs_score) - 1):
                cls_output = self.classifier(
                    torch.cat((paragraphs_score[i], paragraphs_score[i + 1]))
                )
                file_logits.append(self.sigmoid(cls_output).item())

            logits.append(file_logits)

        return logits


    # #make predictions
    def predict(self):
        for paragraphs, key in self.task.paragraphs['test']:
            paragraphs_score = []
            for paragraph in paragraphs:
                sentences = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", paragraph)
                paragraph_score = torch.zeros(self.bert_model.config.hidden_size)
                for sentence in sentences:
                    # print(len(re.split(' |.', sentence)))
                    # print(sentence)
                    if len(re.split(' |.', sentence)) > 512:
                        # print("too long sentence")
                        result = []
                        randomInd = np.random.choice(len(sentence), 505)
                        sortedInd = sorted(randomInd)
                        for ind in sortedInd:
                            result.append(sentence[ind])

                        sentence = ' '.join(result)

                    bert_output = self.bert_model(
                        tokenizer.encode(sentence, return_tensors="pt")
                    )
                    last_hidden_state = bert_output.last_hidden_state
                    sentence_score = torch.sum(last_hidden_state[0], axis=0)
                    paragraph_score = torch.add(paragraph_score, sentence_score)
                paragraph_score = paragraph_score / len(sentences)
                paragraphs_score.append(paragraph_score)

            file_preds = []
            for i in range(len(paragraphs_score) - 1):
                cls_output = self.classifier(torch.cat((paragraphs_score[i], paragraphs_score[i + 1])))
                file_preds.append(round(self.sigmoid(cls_output).item()))

            self.preds[key] = file_preds

        return self.preds


#use predictions to write to solution folder
def writeSolutionFolder(self, task, preds):
    for k, v in self.preds.items():
        fileName = f'solution-problem-{k}.json'
        outputDir = f'/content/solution/BertModel/Task{self.task.task_num}'
        filePath = outputDir + os.sep + fileName # Format of output file is 'solution-problem-*.json'
        v = [int(val) for val in v] #convert Type int64 to int so value is JSON serializable
        key = 'changes'
        dictionaryToWrite = {key : v}
        with open(filePath, "w") as f:
            json.dump(dictionaryToWrite, f)

def print_statistics(self):
    model = load(f"/content/Task{self.task.task_num}.model")
    preds = model.predict()
    self.writeSolutionFolder(self.task.task_num, preds)

    taskSolutions = read_solution_files(
        f'/content/solution/BertModel/Task{self.task.task_num}')
    taskTruth = read_ground_truth_files(
        os.path.join(self.task.datasetValidationFilePath))

    taskMetricsDict = compute_score_multiple_predictions(
            taskTruth, taskSolutions, 'changes', labels=[0, 1])

    for k, v in taskMetricsDict.items():
        write_output(os.path.join(args.output, EV_OUT), f'task{args.taskNum}_{k}', v)

# def predict_authorships(paragraphs):
#   paragraphs_score = []
#   for paragraph in paragraphs:
#     sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
#     paragraph_score = torch.zeros(bert_model.config.hidden_size)
#     for sentence in sentences:
#       # print(tokenizer(sentence))
#       # print(tokenizer.encode(sentence))
#       # print(tokenizer.encode(sentence, return_tensors='pt'))
#       bert_output = bert_model(tokenizer.encode(sentence, return_tensors='pt'))
#       last_hidden_state = bert_output.last_hidden_state
#       sentence_score = torch.sum(last_hidden_state[0], axis = 0)
#       paragraph_score = torch.add(paragraph_score, sentence_score)
#     paragraph_score = paragraph_score/len(sentences)
#     paragraphs_score.append(paragraph_score)

#   outputs = []
#   for i in range(len(paragraphs_score)-1):
#     cls_output = classifier(torch.cat((paragraphs_score[i], paragraphs_score[i+1])))
#     outputs.append(round(sigmoid(cls_output).item()))

#   return outputs

# outputs = predict_authorships(paragraphs)

# print(outputs)
