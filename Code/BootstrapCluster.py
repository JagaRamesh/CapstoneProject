from __future__ import division

import pandas as pd
import numpy as np
import operator

from sklearn import preprocessing
from sklearn.cluster import KMeans
from numpy import unravel_index
from collections import defaultdict

class BootstrapCluster(object):
    '''Class for Bootstraping Times Series data'''

    def __init__(self, n_clusters, n_samples=1000):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.n_clusters = n_clusters
        self.n_samples = n_samples

        self.X = None
        self.rowcount = None
        self.bootstrap_count_matrix = None
        self.cluster_dict = {}
        self.labels = None

    def fit(self,X):
        self.X = X
        self.rowcount = X.shape[0]
        boot_ct_mat = np.zeros((self.rowcount,self.rowcount))
        kmeans = KMeans(n_clusters=self.n_clusters,random_state=0)

        bootstrap = self.bootstrap_indexes()

        for indices in bootstrap:
            X_boot = self.get_bootstrap_data(indices)
            kmeans.fit(X_boot)
            y = kmeans.labels_
            boot_ct_mat = self.update_boot_count(y,boot_ct_mat)

        self.bootstrap_count_matrix = boot_ct_mat
        self.cluster_dict = self.get_clusters()
        self.labels = self.get_labels()

    def bootstrap_indexes(self):
        '''
        Given X points, where axis 1 is considered to delineate points, return
        an array where each row is a set of bootstrap indexes. This can be used as a list of bootstrap indexes as well.
        '''
        return np.random.randint(self.X.shape[1],size=(self.n_samples,self.X.shape[1]))

    def get_bootstrap_data(self,indices):
        '''
        Return dataframe for the indices
        '''
        s1 = []
        for col_ind in indices:
            s2 = self.X.ix[:,col_ind]
            if len(s1) != 0:
                s1 = pd.concat([s1, s2], axis=1)
            else:
                s1 = s2
        return s1

    def update_boot_count(self,y,boot_ct_mat):
        for i in range(self.n_clusters):
            ind = np.where(y==i)[0]
            if len(ind) > 1:
                r=ind[0]
                for j in range(1,len(ind)):
                    c = ind[j]
                    boot_ct_mat[r][c]=boot_ct_mat[r][c]+1
        return boot_ct_mat

    # def get_cluster_count_matrix(self,bootstrap):
    #     '''
    #     Read indices from Bootstrap sample_indices
    #     Run Kmeans , count the no of times the same zipcodes cluster together
    #     Return the count matrix
    #     '''
    #     for indices in bootstrap:
    #         X_boot = bc.get_bootstrap_data(indices)
    #         kmeans.fit(X_boot)
    #         y = kmeans.labels_
    #         boot_ct_mat = bc.update_boot_count(y,boot_ct_mat)
    #     return boot_ct_mat

    def get_isloop(self,d_values,d_keys):
        zip_assigned = len(d_values)
        cl_used = len(d_keys)
        remaining = self.rowcount - zip_assigned
        if (remaining == 0) or (cl_used + remaining <= self.n_clusters):
            return False
        else:
            return True

    def update_dict(self,d,row,col):
        for i,j in d.items():
            if (row in j) or (col in j):
                d[i].add(row)
                d[i].add(col)
        return d

    def get_clusters(self):
        '''
        Read the bootstrap count matrix
        Returns dictionary with key as Clusterid and values as X indices belongs to cluster
        '''
        d = defaultdict(set)
        init = 0
        loop_check = True
        d_values = []
        boot_mat = np.copy(self.bootstrap_count_matrix)

        while loop_check:
            row,col = unravel_index(boot_mat.argmax(), boot_mat.shape)
            if init == 0:
                d[init].add(row)
                d[init].add(col)
                boot_mat[row][col] = 0
                init = 1
            else:
                d_values = [j for i in d.values() for j in i]
                d_keys = [i for i in d.keys()]
                if self.get_isloop(d_values,d_keys):
                    if row==col:
                        print 'Error: Bootstrap Max Argument returns Row & Col as same value. Row : {}, Col : {}'.format(row,col)
                        sys.exit(1)
                    elif (row in d_values) and (col in d_values):
                        boot_mat[row][col] = 0
                    elif (row in d_values) or (col in d_values):
                        d = self.update_dict(d,row,col)
                        boot_mat[row][col] = 0
                    else:
                        dict_max_index=max(d.iteritems(), key=operator.itemgetter(0))[0]
                        d[dict_max_index+1].add(row)
                        d[dict_max_index+1].add(col)
                        boot_mat[row][col] = 0
                else:
                    a = range(self.rowcount)
                    b = d_values
                    c = [ind for ind in a if ind not in b]
                    dict_max_index=max(d.iteritems(), key=operator.itemgetter(0))[0]
                    for i in c:
                        d[dict_max_index+1].add(i)
                        dict_max_index += 1
                    loop_check=False
        return d

    def get_dict_key(self,d,value):
        for key,value_set in d.iteritems():
            if value in value_set:
                return key

    def get_labels(self):
        '''
        Read the dictionary with clusterid as key as X indices as values
        Return Cluster labels
        '''
        d = self.cluster_dict
        labels = []
        for i in range(self.rowcount):
            cluster_value=self.get_dict_key(d,i)
            labels.append(cluster_value)
        return labels
