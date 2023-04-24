import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import re

#first, install the hugging face transformer package in your colab
# !pip install transformers
from transformers import get_linear_schedule_with_warmup
from tokenizers.processors import BertProcessing

# Do not change this line, as it sets the model the model that Hugging Face will load
# If you are interested in what other models are available, you can find the list of model names here:
# https://huggingface.co/transformers/pretrained_models.html
bert_model_name = 'distilbert-base-uncased' 
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



class BertModel():

	def __init__(self, task):
		self.task = task

	def createModel(self):
		print("createModel")
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

		print('after for loop')
		self.NUM_EPOCHS = 1
		print('after num epochs')
		self.bert_model = DistilBertModel.from_pretrained(bert_model_name, num_labels=2)
		print("after x")
		self.classifier = nn.Linear(2*self.bert_model.config.hidden_size, 1)
		self.sigmoid = nn.Sigmoid()
		print("before optimizer")
		self.optimizer = optim.Adam(self.parameters(), lr=0.01)
		print("after optimizer")

		self.train(inputTrainingData, inputLabels, f'models/bert_model/Task{self.task.task_num}.model')
 

	def train(self, inputTrainingData, inputLabels, modelPath):
		if len(inputData) != len(truthLabels):
			raise AssertionError('Input data and ground truth dimension mismatch')
		numDataPoints = len(inputData)

		epoch_loss = 0
		batch_size = 100
		for batch_start in range(0, numDataPoints, batchSize):
			batch_end = batch_start + batch_size
			if batch_end > numDataPoints:
				batch_end = numDataPoints
			source = inputTrainingData[batch_start:batch_end]
			target = inputLabels[batch_start:batch_end]
			self.optimizer.zero_grad()
			output = self.forward(source)
			loss = F.cross_entropy(output, target)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			epoch_loss += loss.item()

		path = os.path.abspath(os.getcwd())
		dump(self, path + os.sep + modelPath)


	def forward(self, inputTrainingData):

		logits = []

		for file in inputTrainingData:
			paragraphs = inputTrainingData[file]

			paragraphs_score = []
			for paragraph in paragraphs:
				sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
				paragraph_score = torch.zeros(bert_model.config.hidden_size)
				for sentence in sentences:
				  # print(tokenizer(sentence))
				  # print(tokenizer.encode(sentence))
				  # print(tokenizer.encode(sentence, return_tensors='pt'))
				  bert_output = bert_model(tokenizer.encode(sentence, return_tensors='pt'))
				  last_hidden_state = bert_output.last_hidden_state
				  sentence_score = torch.sum(last_hidden_state[0], axis = 0)
				  paragraph_score = torch.add(paragraph_score, sentence_score)
				paragraph_score = paragraph_score/len(sentences)
				paragraphs_score.append(paragraph_score)
			  
			file_logits = []
			for i in range(len(paragraphs_score)-1):
				cls_output = classifier(torch.cat((paragraphs_score[i], paragraphs_score[i+1])))
				file_logits.append(sigmoid(cls_output).item())

			logits.append(file_logits)

		return logits

# #make predictions
# def predict(self):


# #use predictions to write to solution folder
# def writeSolutionFolder(self, task, predictedValues):



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