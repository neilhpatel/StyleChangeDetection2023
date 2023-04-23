import argparse
import glob
import json
import os
from itertools import chain
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.sparse import coo_matrix

def read_solution_files(solutions_folder: str) -> dict:
    """
    reads all solution files into dict
    :param solutions_folder: path to folder holding the solution files
    :return: dict of solution files with problem-id as key and file content as value
    """
    solutions = {}
    for solution_file in glob.glob(os.path.join(solutions_folder, 'solution-problem-*.json')):
        with open(solution_file, 'r') as fh:
            curr_solution = json.load(fh)
            solutions[os.path.basename(solution_file)[9:-5]] = curr_solution
    return solutions


def read_ground_truth_files(truth_folder: str) -> dict:
    """
    reads ground truth files into dict
    :param truth_folder: path to folder holding ground truth files
    :return: dict of ground truth files with problem-id as key and file content as value
    """
    truth = {}
    for truth_file in glob.glob(os.path.join(truth_folder, 'truth-problem*.json')):
        with open(truth_file, 'r') as fh:
            curr_truth = json.load(fh)
            truth[os.path.basename(truth_file)[6:-5]] = curr_truth
    return truth

def extract_task_results(truth: dict, solutions: dict, task: str) -> tuple:
    """
    extracts truth and solution values for a given task
    :param truth: dict of all ground truth values with problem-id as key
    :param solutions: dict of all solution values with problem-id as key
    :param task: task for which values are extracted (string, e.g., 'multi-author' or 'changes')
    :return: list of all ground truth values, list of all solution values for given task
    """
    all_solutions = []
    all_truth = []
    for problem_id, truth_instance in sorted(truth.items()):
        if len(truth_instance[task]) != len(solutions[problem_id][task]):
            print(
                f"Solution length for problem {problem_id} is not correct, skipping.")
            continue
        all_truth.append(truth_instance[task])
        all_solutions.append(solutions[problem_id][task])
    return all_truth, all_solutions


def compute_score_multiple_predictions(truth_values: dict, solution_values: dict, key: str, labels: list) -> float:
    """ compute f1 score for list of predictions
    :param labels: labels used for the predictions
    :param truth_values: list of ground truth values for all problem-ids
    :param solution_values: list of solutions for all problem-ids
    :param key: key of solutions to compute score for (=task)
    :return: f1 score
    """
    # extract truth and solution values in suitable format
    truth, solution = extract_task_results(truth_values, solution_values, key)
    # lists have to be flattened first
    return f1_score(list(chain.from_iterable(truth)), list(chain.from_iterable(solution)), average='macro',
                    labels=labels, zero_division=0)


def write_output(filename: str, k: str, v: str):
    """
    print() and write a given measurement to the indicated output file
    :param filename: full path of the file, where to write to
    :param k: the name of the metric
    :param v: the value of the metric
    :return: None
    """
    line = 'measure{{\n  key: "{}"\n  value: "{}"\n}}\n'.format(k, str(v))
    print(line)
    open(filename, "a").write(line)


def main(args, EV_OUT, taskNum):
    if args is None:
        parser = argparse.ArgumentParser(
            description='PAN23 Style Change Detection Task: Evaluator')
        parser.add_argument("-p", "--predictions",
                            help="path to the dir holding the predictions (in a folder for each dataset/task)", required=True)
        parser.add_argument(
            "-t", "--truth", help="path to the dir holding the true labels (in a folder for each dataset/task)", required=True)
        parser.add_argument(
            "-o", "--output", help="path to the dir to write the results to", required=True)
        parser.add_argument(
            "-m", "--model", help="model to evaluate results for", required=True)
        parser.add_argument(
            "-task", "--taskNum", help="task to calculate statistics for", required=True)
        args = parser.parse_args()
        EV_OUT = f'{args.model}.prototext'


    taskSolutions = read_solution_files(
        os.path.join(args.predictions))
    taskTruth = read_ground_truth_files(
        os.path.join(args.truth))

    try:
        taskF1 = compute_score_multiple_predictions(
            taskTruth, taskSolutions, 'changes', labels=[0, 1])
    except KeyError as _:
        taskF1 = None
        print("No solution file found for one or more problems, please check the output. Exiting task 1.")

    # task2_solutions = read_solution_files(
    #     os.path.join(args.predictions))
    # task2_truth = read_ground_truth_files(
    #     os.path.join(args.truth))
    #
    # try:
    #     task2_f1 = compute_score_multiple_predictions(
    #         task2_truth, task2_solutions,  'changes', labels=[0, 1])
    # except KeyError as _:
    #     task2_f1 = None
    #     print("No solution file found for one or more problems, please check the output. Exiting task 2.")
    #
    # task3_solutions = read_solution_files(
    #     os.path.join(args.predictions))
    # task3_truth = read_ground_truth_files(
    #     os.path.join(args.truth))
    # try:
    #     task3_f1 = compute_score_multiple_predictions(
    #         task3_truth, task3_solutions, 'changes', labels=[0, 1])
    # except KeyError as _:
    #     task3_f1 = None
    #     print("No solution file found for one or more problems, please check the output. Exiting task 3.")

    for k, v in {
        f'task{args.taskNum}_f1_score': taskF1,}.items():
        write_output(os.path.join(args.output, EV_OUT), k, v),

def evaluatorMainWrapper(predictionsPath, groundTruthPath, outputPath, model, taskNum):
    parser = argparse.ArgumentParser(description='Evaluation script AV@PAN2023')
    args = parser.parse_args()
    EV_OUT = f'{model}.prototext'
    args.truth = groundTruthPath
    args.predictions = predictionsPath
    args.output = outputPath
    args.taskNum = taskNum
    main(args, EV_OUT, taskNum)


if __name__ == "__main__":
    main(None, None, None)