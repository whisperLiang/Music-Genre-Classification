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

    # Define range of training set sizes to test
    train_sizes = [100, 200, 300, 400, 600, 700, 800, 900]

    # Set up k-fold cross-validation
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Set up lists to hold performance metrics for each training set size
    accuracies = []
    precisions = []
    recalls = []

    for train_size in train_sizes:
        # Set up lists to hold results from each fold
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []

        # Perform cross-validation
        for train_indices, test_indices in kfold.split(dataset):
            # Get the subset of the dataset for the current fold
            train_set = [dataset[i] for i in train_indices[:train_size]]
            test_set = [dataset[i] for i in test_indices]

            # Train and test the model on the current fold
            summaries = train(train_set)
            predictions = predict(summaries, test_set)
            accuracy, precision, recall = evaluate(test_set, predictions)

            # Record results for the current fold
            fold_accuracies.append(accuracy)
            fold_precisions.append(precision)
            fold_recalls.append(recall)

        # Calculate mean performance metrics across all folds for the current training set size
        mean_accuracy = np.mean(fold_accuracies)
        mean_precision = np.mean(fold_precisions)
        mean_recall = np.mean(fold_recalls)

        print(f'Training set size: {train_size}')
        print(f'Accuracy: {np.mean(mean_accuracy):.2f}%')
        print(f'Precision: {np.mean(mean_precision):.2f}%')
        print(f'Recall: {np.mean(mean_recall):.2f}%')

        # Record mean performance metrics for the current training set size
        accuracies.append(mean_accuracy)
        precisions.append(mean_precision)
        recalls.append(mean_recall)

    # Plot performance metrics as a function of training set size
    plt.plot(train_sizes, accuracies, label='Accuracy')
    plt.plot(train_sizes, precisions, label='Precision')
    plt.plot(train_sizes, recalls, label='Recall')
    plt.xlabel('Training Set Size')
    plt.ylabel('Performance (%)')
    plt.title('Performance vs. Training Set Size')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()