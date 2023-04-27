# StyleChangeDetection2023

This is the repository for the 2023 SCD Challenge organizde by PAN. 
Description and details of the challenge can be found on the [PAN website](https://pan.webis.de/clef23/pan23-web/style-change-detection.html).


## Obtaining the Dataset
The dataset can be obtained on [Zenodo](https://zenodo.org/record/7729178#.ZEnUrC-B1Z0). Follow the steps below:
1. Create a Zenodo account and login.
2. Request access to the dataset in the link above and provide details on usage of dataset
3. A private download link to the dataset will be emailed to the provided email address in the form above once access is approved

## Running the Code
First, install all dependencies by running 

```pip install -r requirements.txt```

The code can be run in multiple ways depending on what you want to execute. To execute the entire package, run

```python main.py```

This command will 
1. Read the entire dataset and split the data into train/val/test split for each task
2. Create and train a Naive Character N-Gram Distance-based (CNG-Dist) and Text Compression model for each task using default hyperparameters (see values below).
These models are saved under ```../models/baseline/{modeltype}```
3. Make predictions on the testing dataset for each task by each model with the results being stored in 
```../solution/baseline/{modeltype}/{taskNumber}```
4. Evaluate performance by reading through the predictions and comparing to the ground truth. 
Micro/Macro/Weighted precision, recall, and F1 values are calculated for each model for each task. 
All metric results can be found in ```../results/{modeltype}.prototext``` but weighted metrics were used for the report.

Note that the command above will take time to execute since two models are being created and trained for each task.

### Hyperparameter Finetuning
You can also directly finetune the hyperparameters from the command line when creating CNG-Dist and Text Compression models.

Text Compression has a hyperparameter for character context window size or ppm_order (-ppm). The default value is 5.

CNG-Dist has the following hyperparameters:
- Vocabulary Size: maximum features considered for CNG Dist model ordered by term frequency across the corpus (-vocab_size). The default value is 3000.
- Ngram Size: size of n-grams extracted from text data (-ngram). The default value is 4.
- Iterations: number of times to randomly sample from n-gram model (-iter). The default value is 0.
- Dropout: proportion of features used when calculating similarity (-dropout). The default value is .5.

All hyperparameters are optional command line arguments, so if not provided, the default values will be used. For example, running

```python main.py -dropout .25 -ngram 6 -iter 2 -ppm 8```

will train a Text Compression model with ppm_order = 8 and a CNG-Dist model with ngram = 6, dropout = .25, iterations = 2, and vocab size = 3000.


### Evaluation Metrics
The script to calculate metrics can also be executed directly given a ground truth directory (-t), the model's predictions directory (-p),
an output file path (-o), the task number to focus on (-task), and the model name (-m). The input parameters should be absolute directory paths.
Sample commands are given below:

```python evaluator.py -task 3 --model "Compressor" -t "../StyleChangeDetection2023predicted/CEBaseline" -p "../StyleChangeDetection2023/release" -o "../StyleChangeDetection/results"```

```python evaluator.py -task 3 --model "CNG-Dist" -t "../StyleChangeDetection2023predicted/CEBaseline" -p "../StyleChangeDetection2023/release" -o "../StyleChangeDetection/results"```
