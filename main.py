from Task import Task
from CEBaseline import CEBaseline


if __name__ == "__main__":
    task1 = Task(1)
    task1.runCustomSplitCalcs()
    task1CEBaseline = CEBaseline(task1)
    task1CEBaseline.createModel() #takes ~2 minutes to train model for task 1
    task1CEBaseline.testModel() #takes ~30 seconds to test model for task 1

    task2 = Task(2)
    task2.runCustomSplitCalcs()
    task2CEBaseline = CEBaseline(task2)
    task2CEBaseline.createModel() #takes ~4 minutes to train model for task 2
    task2CEBaseline.testModel() #takes ~80 seconds to test model for task 2

    task3 = Task(3)
    task3.runCustomSplitCalcs()
    task3CEBaseline = CEBaseline(task3)
    task3CEBaseline.createModel() #takes ~2 minutes to train model for task 3
    task3CEBaseline.testModel() #takes ~40 seconds to test model for task 3