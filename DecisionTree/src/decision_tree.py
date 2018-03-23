import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from sklearn import tree
from sklearn import preprocessing
import subprocess
from collections import Counter

# tree node for Decision Tree
class DTreeNode(object):
    def __init__(self, tag, x):
        self.tag = tag;
        self.last_attr = 0;
        self.parent_value = x
        self.feature = ''
        self.lable = ''
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

# Read data set from file
# Random pick 70% for training, 30% for testing.
def data_pre_process():
    # read in data set
    # data = pd.read_csv('./car_evalution.csv')
    # data = pd.read_csv('./mashroom_data.csv')
    data = pd.read_csv('./tic-tac-toe-data.csv')
    data = data[data.columns[0:]]
    if verbose == 1:
        print(data)

    # 70% for training, 30% for testing
    data = data.sample(frac=1)  # shuffle data
    data_training = data.sample(frac=0.7)
    data_test = data.loc[~data.index.isin(data_training.index)]
    print("Training data size: ", data_training.shape[0])
    print("Testing  data size: ", data_test.shape[0])

    if verbose == 1:
        print(data_training)
        print(data_test)

    data_target = data[data.columns[-1:]]
    training_target = data_training[data_training.columns[-1:]]
    test_target = data_test[data_test.columns[-1:]]

    # distribution of classification
    distribution = data_target.groupby(data.columns[-1]).size() / data.shape[0]
    global default_predict
    default_predict = distribution.idxmax()
    train_distribution = training_target.groupby(data_training.columns[-1]).size() / data_training.shape[0]
    test_distribution = test_target.groupby(data_test.columns[-1]).size() / data_test.shape[0]
    if verbose == 1:
        plt.subplot(1, 3, 1)
        distribution.plot(kind='bar', title='Whole Data Set')
        plt.subplot(1, 3, 2)
        train_distribution.plot(kind='bar', title='Training Data Set')
        plt.subplot(1, 3, 3)
        test_distribution.plot(kind='bar', title='Testing Data Set')
        print("Total data set size: ", data.shape[0], " distribution: ", distribution)
        print("Training   set size: ", data_training.shape[0], " distribution: ", train_distribution)
        print("Testing    set size: ", data_test.shape[0], " distribution: ", test_distribution)
        plt.show()
    return data, data_training, data_test;


# Function: calculate Entropy
# Algorithm: entropyS = -SUM(p * log(p,2))
def cal_entropy(data):
    train_target = data[data.columns[-1:]]
    distribution = train_target.groupby(data.columns[-1]).size() / data.shape[0]
    entropyS = 0
    for i in range(distribution.shape[0]):
        entropyS = entropyS - distribution[i] * math.log(distribution[i], 2)
    return entropyS


def fill_missing_data(data):
    num_of_features = data.shape[1] - 1
    for i in range(0, num_of_features):
        if data[data.columns[i]].str.contains('\?').any():
            print("Filling missing data...")
            one_col = data[data.columns[i:i + 1]]
            words = one_col[one_col.columns[0]].unique()
            distribution = one_col.groupby(one_col.columns[0]).size() / one_col.shape[0]
            default_value = distribution.idxmax()
            data_new = data
            data_new[data.columns[i:i + 1]] = data[data.columns[i:i + 1]].replace('?', default_value)
    return data


# Function: build decision tree ID3
# Algorithm: entropyS = -SUM(p * log(p,2))
#            Gain(S,A)= entropyS - SUM(entropySv * Sv/S)
#            (Sv is the subset A == v, ie: Outlook = Sunny)
def buildTree(data, decisionTreeRoot):
    global tree_node_cnt
    global tree_node_tag
    entropyS = cal_entropy(data)

    data = fill_missing_data(data)
    if verbose == 2:
        print('entropy = ', entropyS)

    # The end of building this branch. if entropyS == 0.0:
    if entropyS == 0.0:
        lable = data.iloc[0][data.columns[-1]]
        decisionTreeRoot.last_attr = 1
        decisionTreeRoot.lable = lable
        if verbose == 2:
            print("Done: ", decisionTreeRoot.parent_value, decisionTreeRoot.feature, decisionTreeRoot.lable)
        return

    # calculate Gain for all features
    num_of_features = data.shape[1] - 1  # remove lable
    gain = np.tile(entropyS, num_of_features)
    for i in range(0, num_of_features):
        # Get the values for this feature
        # ie: Outlook: words = ['Sunny', 'Overcast', 'Rain']
        words = data[data.columns[i:i + 1]]
        words = words[words.columns[0]].unique()
        # Loop each value of the feature
        for word in words:
            # only keep the rows with this value for cal entropy
            data_subset = data[data[data.columns[i]] == word]
            propotion = data_subset.shape[0] / data.shape[0]
            entropySv = cal_entropy(data_subset)
            gain[i] = gain[i] - propotion * entropySv
        if verbose == 2:
            print('Gain[', data.columns[i], '] = ', gain[i])

    # max Gain is the new root
    max_index = np.argmax(gain)
    curNode = decisionTreeRoot
    curNode.feature = data.columns[max_index]
    if verbose == 2:
        print("Pick root: ", curNode.feature)

    # list the values of the new root
    words = data[data.columns[max_index:max_index + 1]]
    words = words[words.columns[0]].unique()

    # creat children for new root
    # one child node for each value
    for word in words:
        if verbose == 2:
            print("Pick root branch for: ", curNode.feature, "==", word)
        newNode = DTreeNode(tree_node_tag, word)
        tree_node_tag = tree_node_tag + 1;
        tree_node_cnt = tree_node_cnt + 1;
        curNode.add_child(newNode)
        # update data for building next level of the tree
        # keep the rows for one specific values (ie:'Outlook == Sunny')
        data_subset = data[data[data.columns[max_index]] == word]
        # drop this feature (ie: 'Outlook) column
        data_subset = data_subset.drop(data.columns[max_index], axis=1)
        if data_subset.shape[0] != 0:
            buildTree(data_subset, newNode)

# Function: get lable from Decision Tree
# walk decision tree according the feature, and
# choose the branch according to the value, till
# the leaf node, that's the lable.
def walkTree(test_data, index, root):
    # No more feature, get lable from leaf node
    # ie: High Temperature -> No
    if root.last_attr == 1:
        return root.lable

    # Go down to corresponding path, then check
    # the next feature recursively
    for column in test_data.columns:
        if root.feature == column:
            # Go down to corresponding branch
            branch = test_data.loc[index][column]
            for c in root.children:
                if c.parent_value == branch:
                    # check next feature
                    return walkTree(test_data, index, c)
            break

    # Couldn't find any result
    return default_predict


# Function: add all edges into file
def plotTree_helper(root, fh):
    # leaf node
    if root.lable != '':
        return

    for c in root.children:
        # Construct edge from root to child
        # Fromat:
        #   1. root.parent_value_feature -> child.parent_value_feature
        #   2. root.parent_value         -> child.lable
        line = "_" + str(root.tag) + "_";
        # root node
        if root.parent_value != '':
            line = line + root.parent_value + "_"
        if root.feature != '':
            line = line + root.feature + "_"
        # ->
        line = line[:-1] + "->" + "_" + str(c.tag) + "_"  # remome extra '_'
        # child node
        if c.parent_value != '':
            line = line + c.parent_value + "_"
        if c.feature != '':
            line = line + c.feature + "_"
        if c.lable != '':
            line = line + c.lable + "_"
        # ;
        line = line[:-1] + ";"  # Remove extra '_'
        fh.write(line)
        plotTree_helper(c, fh)


# Function: plotTree in dot format
# open a file, write all edges with dot format
# then plot to .png
def plotTree(decisionTreeRoot, file_name):
    dot_file_name = file_name + ".dot"
    fh = open(dot_file_name, "w")
    fh.write("digraph G {", )
    plotTree_helper(decisionTreeRoot, fh)
    fh.write("}", )
    fh.close()
    png_file_name = file_name + ".png"
    subprocess.call(["dot.exe", "-Tpng", dot_file_name, "-o", png_file_name])



# Function: use the Decision Tree to predict
# Add one column 'Prediction' in test_data
# Call Decision Tree to get the prediction
# add the prediction lable to this column
# return new table
def predict(test_data, decisionTreeRoot):
    predict_result = fill_missing_data(test_data)
    predict_result['Prediction'] = ""
    for i, row in predict_result.iterrows():
        lable = walkTree(test_data, i, decisionTreeRoot)
        predict_result.loc[i, 'Prediction'] = lable
    return predict_result


def predict_report(data_test):
    data_test_wo_cls = data_test.drop(data_test.columns[-1], axis=1)
    predict_result = predict(data_test_wo_cls, decisionTreeRoot)
    data_test.to_csv('./test_golden.csv', index=True)
    predict_result.to_csv('./predict.csv', index=True)

    golden_target = np.ravel(data_test[data_test.columns[-1]])
    predict_target = np.ravel(predict_result[predict_result.columns[-1]])
    if verbose == 2:
        print("Golden target: ", golden_target)
        print("Predicted result: ", predict_target)
    test_correct_cnt = sum(golden_target == predict_target)
    test_total_cnt = len(golden_target)
    correct_percent = test_correct_cnt / test_total_cnt * 100
    return test_correct_cnt, test_total_cnt, correct_percent

def prune_tree(test_data, root):
    global tree_node_cnt
    global rmv_candidate
    global prune_threshold
    print("------------------- ")
    print("Loop to prune tree, current tree node cnt = ", tree_node_cnt, "  New threshold: ", prune_threshold)
    find_remove_list(test_data, root)
    while rmv_candidate != None:
        print ("Remove node: ", rmv_candidate.tag)
        rmv_candidate.last_attr = 1;
        tree_node_cnt = tree_node_cnt - 1
        rmv_candidate = None
        print ("------------------- ")
        print ("Loop to prune tree, current tree node cnt = ", tree_node_cnt, "  New threshold: ", prune_threshold)
        find_remove_list(test_data, root)

def remove_node(root):
    global tree_node_cnt
    global tree_node_tag
    if root.lable != '' :
        root.last_attr = 1;
    else :
        temp_lable = []
        for c in root.children:
            temp_lable.append(c.lable)
        most_common = Counter(temp_lable).most_common(1)[0][0]
        root.last_attr = 1
        root.lable = most_common
    print(root.lable)

def find_remove_list(test_data, root):
    global rmv_candidate
    global prune_threshold
    remove = True
    for c in root.children:
        if c.last_attr == 0:
            find_remove_list(test_data, c)
            remove = False
    if remove:
        print("try to remove: ", root.tag)
        remove_node(root)
        new_correct_cnt, new_total_cnt, new_correct_percent = predict_report(data_test)
        print("    Correct prediction count on testing data: ", new_correct_cnt)
        # put the node back
        root.last_attr = 0
        if new_correct_cnt >= prune_threshold:
            rmv_candidate = root
            prune_threshold = new_correct_cnt



verbose = 0
tree_node_cnt = 0  # count tree node except leaf node. leaf node is the label
tree_node_tag = 0 # count every tree node including leaf node
default_predict = ""

print("Pre process data: Shuffle data. Random pick 70% for training, the rest 30% for testing.")
print(" ")
data, data_training, data_test = data_pre_process()
decisionTreeRoot = DTreeNode(tree_node_tag, '')
tree_node_cnt = tree_node_cnt + 1
tree_node_tag = tree_node_tag + 1

print("Build ID3 decision tree.")
print(" ")
buildTree(data_training, decisionTreeRoot)

print("How many tree node: ", tree_node_cnt)
print(" ")
plotTree(decisionTreeRoot, "beforePrune")

print("Use test data to predict: ")
test_correct_cnt, test_total_cnt, correct_percent = predict_report(data_test)
print("    Correct prediction count on testing data: ", test_correct_cnt)
print("    Wrong   prediction count on testing data: ", test_total_cnt - test_correct_cnt)
print("    Total   prediction count on testing data: ", test_total_cnt)
print("    Correct percentage       on testing data: ", correct_percent, "%")
print("--")
print("Output golden    result into file ./test_golden.csv")
print("Output predicted result into file ./predicted.csv")

print("Use training data to predict: ")
tr_correct_cnt, tr_total_cnt, tr_correct_percent = predict_report(data_training)
print("    Correct prediction count on training data: ", tr_correct_cnt)
print("    Wrong   prediction count on training data: ", tr_total_cnt - tr_correct_cnt)
print("    Total prediction count   on training data: ", tr_total_cnt)
print("    Correct percentage       on training data: ", tr_correct_percent, "%")

entire_correct_percent = (tr_correct_cnt + test_correct_cnt) / (test_total_cnt + tr_total_cnt) * 100
print("    -----")
print("    Correct percentage       on whole    data: ", entire_correct_percent, "%")

prune_threshold = test_correct_cnt
rmv_candidate = None
prune_tree(data_test, decisionTreeRoot)
print("Tree node cnt after prune = ", tree_node_cnt)
plotTree(decisionTreeRoot, "afterPrune")


print("Use test data to re-predict after pruning the tree: ")
prune_test_correct_cnt, prune_test_total_cnt, prune_correct_percent = predict_report(data_test)
print("    Correct prediction count on testing data after prune: ", prune_test_correct_cnt)
print("    Wrong   prediction count on testing data after prune: ", prune_test_total_cnt - prune_test_correct_cnt)
print("    Total   prediction count on testing data after prune: ", prune_test_total_cnt)
print("    Correct percentage       on testing data after prune: ", prune_correct_percent, "%")
print("--")
print("Output golden    result into file ./test_golden.csv")
print("Output predicted result into file ./predicted.csv")

print("Use training data to re-predict after pruning the tree: ")
prune_tr_correct_cnt, prune_tr_total_cnt, prune_tr_correct_percent = predict_report(data_training)
print("    Correct prediction count on training data after prune: ", prune_tr_correct_cnt)
print("    Wrong   prediction count on training data after prune: ", prune_tr_total_cnt - prune_tr_correct_cnt)
print("    Total prediction count   on training data after prune: ", prune_tr_total_cnt)
print("    Correct percentage       on training data after prune: ", prune_tr_correct_percent, "%")

prune_entire_correct_percent = (prune_tr_correct_cnt + prune_test_correct_cnt) / (prune_test_total_cnt + prune_tr_total_cnt) * 100
print("    -----")
print("    Correct percentage       on whole    data after prune: ", prune_entire_correct_percent, "%")

print(" ")
print(" ")
print(" Compare before prune - after prune")
print(" --------- ")
print("    Correct percentage       on testing  data befor prune: ", correct_percent, "%")
print("    Correct percentage       on training data befor prune: ", tr_correct_percent, "%")
print("    Correct percentage       on whole    data befor prune: ", entire_correct_percent, "%")

print("    Correct percentage       on testing  data after prune: ", prune_correct_percent, "%")
print("    Correct percentage       on training data after prune: ", prune_tr_correct_percent, "%")
print("    Correct percentage       on whole    data after prune: ", prune_entire_correct_percent, "%")