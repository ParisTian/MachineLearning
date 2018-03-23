import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import sys
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
def pre_process(data):
    verbose = 0
    data_attr = data[data.columns[:-1]]
    data_golden_label = pd.DataFrame(data[data.columns[-1]])
    words = data_golden_label[data_golden_label.columns[0]].unique()
    # change the golden label to id: 0,1,2...
    id = 0
    for word in words:
        data_golden_label = data_golden_label.replace(word, id)
        id = id + 1
    if verbose == 1:
        print (data_golden_label)
    data_label = pd.DataFrame(data_attr.index, columns = ['C'])
    # each point is a cluster
    data_label['C'] = data_label.index
    return data_attr, data_label, data_golden_label

def post_process(data_label):
    global data_golden_label
    array_3 = [[0, 1, 2], [0, 2, 1], \
               [1, 0, 2], [1, 2, 0], \
               [2, 0, 1], [2, 1, 0]]
    array_4 = [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [0, 2, 1, 3], [0, 3, 2, 1], [0, 3, 1, 2], \
               [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 3, 0], [1, 2, 0, 3], [1, 3, 2, 0], [1, 3, 0, 2], \
               [2, 1, 0, 3], [2, 1, 3, 0], [2, 0, 3, 1], [2, 0, 1, 3], [2, 3, 0, 1], [2, 3, 1, 0], \
               [3, 1, 2, 0], [3, 1, 0, 2], [3, 2, 0, 1], [3, 2, 1, 0], [3, 0, 2, 1], [3, 0, 1, 2]]
    words = data_label[data_label.columns[0]].unique()
    if words.size == 3:
        array = array_3
    elif words.size == 4:
        array = array_4
    # change cluster id to 0, 1...
    id = -1
    # print (np.ravel(data_label))
    for word in words:
        data_label = data_label.replace(word, id)
        id = id - 1
    # print (np.ravel(data_label))

    best_accuracy = 0
    best_order = 0
    for i in range(0, len(array)):
        temp_label = data_label
        old_id = -1
        for j in range(0, len(array[0])):
            temp_label = temp_label.replace(old_id, array[i][j])
            old_id = old_id - 1
        temp_accuracy = sum(np.ravel(data_golden_label) == np.ravel(temp_label))
        if verbose == 1:
            print ("accu for: ", i, " ", temp_accuracy)
        if temp_accuracy > best_accuracy:
            best_accuracy = temp_accuracy
            best_order = i

    old_id = -1
    for j in range(0, len(array[0])):
        data_label = data_label.replace(old_id, array[best_order][j])
        old_id = old_id - 1

    data_attr['C'] = data_label['C']
    if verbose == 1:
        print(data_attr)

    golden_cluster = np.ravel(data_golden_label)
    my_cluster = np.ravel(data_label)
    if verbose == 1:
        print(np.ravel(my_cluster))
        print(np.ravel(golden_cluster))
    my_accuracy = sum(golden_cluster == my_cluster)
    return my_accuracy, data_label

def linkage_clustering(data_attr, data_label, cluster_alg):
    verbose = 0
    # calculate the distance
    dist = pdist(data_attr, 'euclidean')
    df_dist = squareform(dist)

    while True:
        # replace clustering information
        data_attr['C'] = data_label['C']
        how_many_clusters = data_label[data_label.columns[0]].unique().size
        if verbose == 1:
            print("how_many_clusters: ", how_many_clusters)
        if how_many_clusters == k:
            return data_label
        if verbose == 1:
            print(data_attr)
            print(df_dist)
        # find the min value index
        min_value = sys.maxsize
        for i in range(0, len(df_dist)):
            for j in range(i + 1, len(df_dist)):
                if df_dist[i][j] == 0:
                    min_x = i
                    min_y = j
                    min_value = 0
                    break
                elif df_dist[i][j] == -1:
                    continue
                elif df_dist[i][j] < min_value:
                    min_x = i
                    min_y = j
                    min_value = df_dist[i][j]
            if min_value == 0:
                break
        # union clusters
        # print (min_value, " ")
        cluster_a = data_label.C[min_x]
        cluster_b = data_label.C[min_y]
        if verbose == 1:
            print("Merger cluster: ", cluster_a, " with cluster: ", cluster_b, " New cluster name: ", cluster_a)

        data_label = pd.DataFrame(data_label.C.replace(cluster_b, cluster_a))
        if verbose == 1:
            print(data_label)

        # clear the distance for two clusters, they merge together, dist = -1
        array = np.ravel(data_label)
        for i in range(0, array.size):
            if array[i] == cluster_a:
                for j in range(i+1, array.size):
                    if array[i] == cluster_a and array[j] == cluster_a:
                        df_dist[i][j] = -1
                        df_dist[j][i] = -1
                        if verbose == 1:
                            print("-----------")
                            print("Clear to -1: (", i, ",", j, ") (", j, ",", i, ")")

        for i in range(0, array.size):
            if cluster_alg == "single":
                next_value = sys.maxsize
            elif cluster_alg == "complete":
                next_value = 0;

            if array[i] != cluster_a:
                for j in range(0, array.size):
                    if array[j] == cluster_a:
                        if cluster_alg == "single":
                            next_value = min(next_value, df_dist[i][j])
                        elif cluster_alg == "complete":
                            next_value = max(next_value, df_dist[i][j])

                for j in range(0, array.size):
                    if array[j] == cluster_a:
                        df_dist[i][j] = next_value
                        df_dist[j][i] = next_value
                        if verbose == 1:
                            print("Take next value: ", next_value, "(", i, ",", j, ") (", j, ",", i, ")")

def average_linkage_clustering(data_attr, data_label, cluster_alg):
    verbose = 0
    # calculate the distance
    dist = pdist(data_attr, 'euclidean')
    df_dist = squareform(dist)

    while True:
        # replace clustering information
        data_attr['C'] = data_label['C']
        how_many_clusters = data_label[data_label.columns[0]].unique().size
        if verbose == 1:
            print("how_many_clusters: ", how_many_clusters)
        if how_many_clusters == k:
            return data_label
        if verbose == 1:
            print(data_attr)
            print(df_dist)
        # find the min value index
        min_value = sys.maxsize
        for i in range(0, len(df_dist)):
            for j in range(i + 1, len(df_dist)):
                if df_dist[i][j] == 0:
                    min_x = i
                    min_y = j
                    min_value = 0
                    break
                elif df_dist[i][j] == -1:
                    continue
                elif df_dist[i][j] < min_value:
                    min_x = i
                    min_y = j
                    min_value = df_dist[i][j]
            if min_value == 0:
                break
        # union clusters
        # print(min_value)
        cluster_a = data_label.C[min_x]
        cluster_b = data_label.C[min_y]
        if verbose == 1:
            print("Merger cluster: ", cluster_a, " with cluster: ", cluster_b, " New cluster name: ", cluster_a)

        data_label = pd.DataFrame(data_label.C.replace(cluster_b, cluster_a))
        if verbose == 1:
            print(data_label)

        # clear the distance for two clusters, they merge together, dist = -1
        array = np.ravel(data_label)
        for i in range(0, array.size):
            if array[i] == cluster_a:
                for j in range(i+1, array.size):
                    if array[i] == cluster_a and data_label.C[j] == cluster_a:
                        df_dist[i][j] = -1
                        df_dist[j][i] = -1
                        if verbose == 1:
                            print("-----------")
                            print("Clear to -1: (", i, ",", j, ") (", j, ",", i, ")")

        clusters_list = data_label[data_label.columns[0]].unique()

        for c in range(0, clusters_list.size):
            if clusters_list[c] != cluster_a:
                avg_value = 0
                avg_cnt = 0
                for i in range(0, array.size):
                    if array[i] == clusters_list[c]:
                        for j in range(0, array.size):
                            if array[j] == cluster_a:
                                if i > j:
                                    temp_dist = df_dist[i][j]
                                else:
                                    temp_dist = df_dist[j][i]
                                avg_value = avg_value + temp_dist
                                avg_cnt = avg_cnt + 1
                avg_value = avg_value / avg_cnt;
                for i in range(0, array.size):
                    if array[i] == clusters_list[c]:
                        for j in range(0, array.size):
                            if array[j] == cluster_a:
                                if i < j:
                                    df_dist[i][j] = avg_value
                                    if verbose == 1:
                                        print("Take avg value: ", avg_value, "(", i, ",", j, ")")
                                else:
                                    df_dist[j][i] = avg_value
                                    if verbose == 1:
                                        print("Take avg value: ", avg_value, "(", j, ",", i, ")")


def lloyd_clustering(data_attr, data_label):
    # Randome pick k centroids
    verbose = 0
    data_attr_label = data_attr
    data_attr_label['C'] = data_label['C']
    pre_centroids = data_attr_label.sample(n=k)
    array = np.arange(k)
    pre_centroids['C'] = array
    #print("List of centroids for Lloyd's : \n", pre_centroids)

    # Clustering points to the nearest centroid
    return cluster_to_nearest_centroid(pre_centroids, data_attr_label, data_attr, data_label)

def cluster_to_nearest_centroid(pre_centroids, data_attr_label, data_attr, data_label):
    verbose = 0
    array = np.arange(k)
    while True:
        if verbose == 1:
            print("current centroids", pre_centroids)

        # assign each point to its cloest center
        for i, row in data_attr_label.iterrows():
            point = np.ravel(row)
            point = point[:-1]
            min_dist = sys.maxsize

            for j, centroid in pre_centroids.iterrows():
                cent = np.ravel(centroid)
                cent = cent[:-1]
                cur_dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point, cent)]))
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    data_attr_label.loc[i, 'C'] = centroid['C']
        if verbose == 1:
            print("after assign each point to its cloest center: ", data_attr_label)
            # calculate the centriod
        new_centroids = data_attr.groupby('C').mean()
        new_cent_array = new_centroids.values
        pre_cent_array = pre_centroids[pre_centroids.columns[:-1]].values
        if (new_cent_array == pre_cent_array).all():
            data_label['C'] = data_attr_label['C']
            return data_label
        else:
            pre_centroids = new_centroids
            pre_centroids['C'] = array


def kmean_pp_clustering(data_attr, data_label):
    # Random pick the first centroid, then find the furthest as the rest
    verbose = 0
    data_attr_label = data_attr
    dist = pdist(data_attr, 'euclidean')
    df_dist = squareform(dist)
    df_dist = np.power(df_dist, 2)
    data_attr_label['C'] = data_label['C']
    # random pick 1st center
    pre_centroids = data_attr_label.sample(n=1)
    which_center = 0
    index = pre_centroids.index[which_center]
    # pick the rest centers
    distance_to_center = df_dist[index]
    for c in range(0, k - 1):
        norm_dist = [i / sum(distance_to_center) for i in distance_to_center]
        new_center = np.random.choice(data_attr_label.index, 1, p=norm_dist)
        for i in range(0, distance_to_center.size):
            distance_to_center[i] = min(distance_to_center[i], df_dist[new_center[0]][i])
        pre_centroids.loc[data_attr_label.index[new_center[0]]] = data_attr_label.loc[new_center[0]]
    array = np.arange(k)
    pre_centroids['C'] = array
    #print("List of centroids for K-mean++: \n", pre_centroids)

    # Clustering points to the nearest centroid
    return cluster_to_nearest_centroid(pre_centroids, data_attr_label, data_attr, data_label)

def hammingDistance (golden_cluster, my_cluster):
    #print ("Golden cluster: \n", golden_cluster)
    #print ("My cluster: \n", my_cluster)
    diff_count = 0
    size = len(golden_cluster)
    #print ("cluster size: ", size, "my cluster size: ", len(my_cluster))
    for i in range (0, size-1) :
        for j in range (i+1, size) :
            if ((golden_cluster[i] == golden_cluster[j]) and (my_cluster[i] != my_cluster[j])) or \
                    ((golden_cluster[i] != golden_cluster[j]) and (my_cluster[i] == my_cluster[j])):
                diff_count = diff_count + 1
    total_count = (size * (size - 1) / 2)
    print ("different: ", diff_count, " total cnt: ", total_count)
    hamming_distance = diff_count / total_count
    return hamming_distance

def pre_plot (data):
    global which_data_set
    x = 2
    if which_data_set == 1:
        y = 2
        end = 4
    elif which_data_set == 2:
        y = 4
        end = 7
    elif which_data_set == 3:
        y = 3
        end = 5
    for id in range(0,end):
        plt.subplot(x, y, id + 1)
        plt.scatter(data[data.columns[id]], data[data.columns[-1]], color='red')
    plt.show()


def post_plot (data, data_golden_label, data_my_label):
    global which_data_set
    x = 2
    if which_data_set == 1:
        y = 2
        end = 4
    elif which_data_set == 2:
        y = 4
        end = 7
    elif which_data_set == 3:
        y = 3
        end = 5
    for id in range(0,end):
        plt.subplot(x, y, id + 1)
        plt.scatter(data[data.columns[id]], data_golden_label, color='red', marker = '_')
        plt.scatter(data[data.columns[id]], data_my_label, color='green', marker = '|')
    plt.show()

######################
#####################
verbose = 0
plot = 1
# 1: iris    2: seed    3: learning_model
which_data_set = 1
normaliztion_cfg = 0
if which_data_set == 1:
    k    = 3
    data = pd.read_csv('./Iris.csv')
elif which_data_set == 2:
    k    = 3
    data = pd.read_csv('./seed.csv')
else:
    k    = 4
    data = pd.read_csv('./learning_model.csv')

if plot == 1:
    pre_plot(data)
if normaliztion_cfg == 1:
    min_max_scale = preprocessing.MinMaxScaler()
    scaled = min_max_scale.fit_transform(data[data.columns[:-1]])
    data_normalized = pd.DataFrame(scaled)
    data[data.columns[:-1]] = data_normalized

#data = data.sample(n = 10)
data_attr, data_label, data_golden_label = pre_process(data)
data_sl_label = linkage_clustering(data_attr, data_label, "single")
my_sl_accuracy, data_sl_label = post_process(data_sl_label)
#print ("Single linkage outcome: \n", np.ravel(data_sl_label))
my_sl_ham_dist = hammingDistance(np.ravel(data_golden_label), np.ravel(data_sl_label))
# print(np.ravel(data_sl_label));
print("My SL    accuracy: ", my_sl_accuracy, " out of ", data_golden_label.shape[0], " :: %3.2f" %
      (my_sl_accuracy / data_golden_label.shape[0] * 100), "%", " Hamming Distance: %.4f" % my_sl_ham_dist)
if plot == 1:
    post_plot(data, data_golden_label, data_sl_label)

data_attr, data_label, data_golden_label = pre_process(data)
data_al_label = average_linkage_clustering(data_attr, data_label, "average")
my_al_accuracy, data_al_label = post_process(data_al_label)
#print ("Average linkage outcome: \n", np.ravel(data_al_label))
my_al_ham_dist = hammingDistance(np.ravel(data_golden_label), np.ravel(data_al_label))
print("My AL    accuracy: ", my_al_accuracy, " out of ", data_golden_label.shape[0], " :: %.2f" %
      (my_al_accuracy / data_golden_label.shape[0] * 100), "%", " Hamming Distance: %.4f" % my_al_ham_dist)
if plot == 1:
    post_plot(data, data_golden_label, data_al_label)

data_attr, data_label, data_golden_label = pre_process(data)
data_cl_label = linkage_clustering(data_attr, data_label, "complete")
#print ("Complete outcome: \n", np.ravel(data_cl_label))
my_cl_accuracy, data_cl_label = post_process(data_cl_label)
#print ("Final Complete outcome: \n", np.ravel(data_cl_label))

#print ("Complete linkage outcome: \n", np.ravel(data_cl_label))
my_cl_ham_dist = hammingDistance(np.ravel(data_golden_label), np.ravel(data_cl_label))
print("My CL    accuracy: ", my_cl_accuracy, " out of ", data_golden_label.shape[0], " :: %.2f" %
      (my_cl_accuracy / data_golden_label.shape[0] * 100), "%", " Hamming Distance: %.4f" % my_cl_ham_dist)
if plot == 1:
    post_plot(data, data_golden_label, data_cl_label)

data_ll_label = lloyd_clustering(data_attr, data_label)
my_ll_accuracy, data_ll_label = post_process(data_ll_label)
my_ll_ham_dist = hammingDistance(np.ravel(data_golden_label), np.ravel(data_ll_label))
print("My Lloyd accuracy: ", my_ll_accuracy, " out of ", data_golden_label.shape[0], " :: %.2f" %
      (my_ll_accuracy / data_golden_label.shape[0] * 100), "%", " Hamming Distance: %.4f" % my_ll_ham_dist)
if plot == 1:
    post_plot(data, data_golden_label, data_ll_label)

data_kmpp_label = kmean_pp_clustering(data_attr, data_label)
my_kmpp_accuracy, data_kmpp_label = post_process(data_kmpp_label)
my_kmpp_ham_dist = hammingDistance(np.ravel(data_golden_label), np.ravel(data_kmpp_label))
print("My KMPP  accuracy: ", my_kmpp_accuracy, " out of ", data_golden_label.shape[0], " :: %3.2f" %
      (my_kmpp_accuracy / data_golden_label.shape[0] * 100), "%", " Hamming Distance: %.4f" % my_kmpp_ham_dist)
if plot == 1:
    post_plot(data, data_golden_label, data_kmpp_label)


#from scipy.spatial.distance import hamming
# This hamming distance is not the same algorithm
#print("Golden hamming for single  : \n", hamming(np.ravel(data_sl_label), np.ravel(data_golden_label)))
#print("Golden hamming for average : \n", hamming(data_golden_label, data_al_label))
#print("Golden hamming for complete: \n", hamming(data_golden_label, data_cl_label))
#print("Golden hamming for Lloyd   : \n", hamming(data_golden_label, data_ll_label))
#print("Golden hamming for KmeanPP : \n", hamming(data_golden_label, data_kmpp_label))

#from scipy.cluster.hierarchy import dendrogram, linkage

#data_attr, data_label, data_golden_label = pre_process(data)
#dist = pdist(data_attr, 'euclidean')
#Y = linkage(dist, 'complete')
#print ("complete from lib: \n")
#print (Y)

#data_attr, data_label, data_golden_label = pre_process(data)
#dist = pdist(data_attr, 'euclidean')
#Y = linkage(dist, 'average')
#print ("Average from lib: \n")
#print (Y)

#data_attr, data_label, data_golden_label = pre_process(data)
#dist = pdist(data_attr, 'euclidean')
#Y = linkage(dist, 'single')
#print ("Single from lib: \n")
#print (Y)