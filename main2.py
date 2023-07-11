import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.model_selection import KFold

def preprocess(filename):
    df = pd.read_csv(filename)
    # 删除无用的列
    df.drop(columns=['filename'], inplace=True)
    dataset = df.values.tolist()

    # 将字符串类型的类别转换为整型
    labels = {}
    for row in dataset:
        label = row[-1]
        if label not in labels:
            labels[label] = len(labels)
        row[-1] = labels[label]

    return dataset

def train(train_set):
    # 分别计算每个类别下每个属性的均值和标准差
    summaries = {}
    for i in range(len(train_set)):
        row = train_set[i]
        class_value = row[-1]
        if class_value not in summaries:
            summaries[class_value] = []
        for j in range(len(row)-1):
            if len(summaries[class_value]) == j:
                summaries[class_value].append([])
            summaries[class_value][j].append(row[j])
    for class_value, attribute_summaries in summaries.items():
        for i in range(len(attribute_summaries)):
            attribute_summaries[i] = (sum(attribute_summaries[i])/float(len(attribute_summaries[i])), 
                                      math.sqrt(sum([pow(x - attribute_summaries[i][0], 2) for x in attribute_summaries[i]]) / float(len(attribute_summaries[i]) - 1)))
    
    return summaries

def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities

def predict(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        probabilities = calculate_class_probabilities(summaries, test_set[i])
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        predictions.append(best_label)
    return predictions

def evaluate(test_set, predictions):
    correct = 0
    tp, fp, fn = 0, 0, 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
            if test_set[i][-1] == 1:
                tp += 1
        else:
            if test_set[i][-1] == 1:
                fn += 1
            else:
                fp += 1
    accuracy = (correct / float(len(test_set))) * 100.0
    precision = (tp / float(tp + fp)) * 100.0
    recall = (tp / float(tp + fn)) * 100.0
    return accuracy, precision, recall

def main():
    train_fname = 'gztan_train.csv'
    test_fname = 'gztan_test.csv'
    train_set = preprocess(train_fname)
    test_set = preprocess(test_fname)
    dataset = train_set + test_set

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Define a range of training set sizes to test
    # train_sizes = [50, 100, 200, 400, 600, 800]
    train_sizes = [100, 200, 300, 400, 600, 700, 800, 900]

    for size in train_sizes:
        acc_scores, prec_scores, rec_scores = [], [], []
        for train_index, test_index in kf.split(dataset):
            train_set = [dataset[i] for i in train_index[:size]]
            test_set = [dataset[i] for i in test_index]
            summaries = train(train_set)
            predictions = predict(summaries, test_set)
            accuracy, precision, recall = evaluate(test_set, predictions)
            acc_scores.append(accuracy)
            prec_scores.append(precision)
            rec_scores.append(recall)
        print(f'Training set size: {size}')
        print(f'Accuracy: {np.mean(acc_scores):.2f}%')
        print(f'Precision: {np.mean(prec_scores):.2f}%')
        print(f'Recall: {np.mean(rec_scores):.2f}%')


if __name__ == "__main__":
    main()