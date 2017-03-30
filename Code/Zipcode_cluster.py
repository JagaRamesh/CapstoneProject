import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from __future__ import division
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score
from BootstrapCluster import BootstrapCluster

def read_clean_data():
    '''
    Read data from csv
    Return cleaned up data as dataframe
    '''
    datafile = './data/Zip_Zhvi_AllHomes_orig.csv'
    df = pd.read_csv(datafile)

    ## Create dataframe only for bayarea zipcodes
    df['Bayarea']='N'
    df.ix[df.CountyName.isin(['San Francisco', 'Contra Costa', 'Alameda', 'Napa', 'Solano',
       'Santa Clara', 'San Mateo', 'Sonoma', 'Marin']),'Bayarea']='Y'
    df_bayarea = df[df.Bayarea=='Y']

    ## Reset index and take data from year 2007 to 2017
    df_bayarea.reset_index(drop=True, inplace=True)
    df_ba_2007_2017 = df_bayarea.ix[:,range(0,6)+range(52+(12*7),258)]

    ## Delete unwanted columns and rename columns to relevant name
    del df_ba_2007_2017['Bayarea']
    del df_ba_2007_2017['RegionID']
    df_ba_2007_2017.rename(columns={'RegionName': 'Zipcode'}, inplace=True)

    ## Get rid of zipcodes with null values
    df_ba_2007_2017 = df_ba_2007_2017[~df_ba_2007_2017.Zipcode.isin([94606,94621,94108,94542,94574,94515,95441])]

    #### Check if any of the values are null
    if df_ba_2007_2017.isnull().values.any():
        print "ERROR : NULL VALUES IN CLEANED UP DATA"
        sys.exit(1)

    ## interpolate value for zipcode 94612
    #df_ba_2007_2017.ix[df_ba_2000_2017.Zipcode==94612,'2000-01'] =    df_ba_2000_2017['2000-02'][df_ba_2000_2017.Zipcode==94612].values

    # write clean data to csv file
    #df_bayarea.to_csv('./data/zillow_bayarea.csv', sep=',',index=False)

    return df_ba_2007_2017

def get_X(df):
    '''
    Read dataframe
    Return dataframe with index as Zipcode and housing price as data
    '''
    X = df.ix[:,[0]+range(5,df.shape[1])]
    X.set_index('Zipcode',inplace=True)
    return X

def get_QX(df):
    X = get_X(df)
    XT = get_XT(X)
    X = XT.resample('Q').astype(int).T
    return X

def get_pct_change(X):
    XT = get_XT(X)
    XX = XT.pct_change().T*100
    XX = XX.ix[:,1:]
    return XX

def get_XT(X):
    '''
    Read dataframe
    Transpose and set index as Month-year
    Return Transposed dataframe
    '''
    XT = X.T
    XT.index = pd.to_datetime(XT.index)
    return XT

def assign_cluster(df,X,cluster_col,no_clusters):
    '''
    Read dataframe & X
    Apply Kmeans and add cluster labels to dataframe
    '''
    kmeans = KMeans(n_clusters=no_clusters,random_state=0)
    kmeans.fit(X)
    df[cluster_col] = kmeans.labels_
    return df

def plot_clusters(df,cluster_col,X,no_clusters,ylim=1):
    '''
    Plot Clustered Zipcodes and its housing price trend
    Each cluster is plotted in the subplot
    4 subplots in a row (ncols=4)
    '''
    if no_clusters % 4 == 0:
        no_rows = no_clusters/4

    else:
        no_rows = (no_clusters//4) + 1
    no_cols = 4
    fig_wid = no_cols * 4
    fig_ht = no_rows * 4

    fig, ax = plt.subplots(nrows=no_rows, ncols=no_cols, figsize=(fig_wid, fig_ht))
    row = 0
    col = -1
    XT = get_XT(X)

    for cluster in range(no_clusters):
        if col == no_cols - 1:
            row = row+1
            col = 0
        else:
            col = col + 1

        for zipcode in df[['Zipcode']][df[cluster_col]==cluster].values:
            zc = int(zipcode)
            ts = pd.Series(XT[zc])/1000000
            if no_rows > 1:
                ax[row,col].set_ylabel("(in millions)")
                plt.setp(ax[row,col].get_xticklabels(), rotation=30, horizontalalignment='right')
                ax[row,col].set_ylim(0, ylim)
                ax[row,col].plot(ts)
            else:
                ax[col].set_ylabel("(in millions)")
                plt.setp(ax[col].get_xticklabels(), rotation=30, horizontalalignment='right')
                ax[col].set_ylim(0, ylim)
                ax[col].plot(ts)
    plt.tight_layout()

def plot_cluster(df,cluster_col,X,clusterid):
    '''
    Plot the housing price data for all zipcodes
    in a particular cluster
    '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    XT = get_XT(X)
    for zipcode,city,county in df[['Zipcode','City','CountyName']][df[cluster_col]==clusterid].values:
        zc = int(zipcode)
        ts = pd.Series(XT[zc])/1000000
        ax.set_ylabel("(in millions)")
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

        ax.plot(ts,label=str(zipcode)+' | '+city+' | '+county)
        plt.legend(bbox_to_anchor=(1, 1), loc=2,prop={'size':6})

def wcss(X,label,centroids,k):
    '''
    Sum squared euclidean distance of all points to their cluster center
    '''

    sse = 0
    sse_clus = 0
    sse_list = []

    for cl_no in range(k):
        ind = np.where(label==cl_no)[0]
        XA = X.values[ind]
        XB = np.matrix(centroids[cl_no])
        sse_clus = np.sum(cdist(XA, XB)**2)
        sse_list.append(sse_clus)
        sse = sse + sse_clus
    return sse,range(k),sse_list

def plot_wcss_silhouette_score(X,min_k, max_k):
    '''
    Plot WCSS (Within cluster Sum of square errors) Vs K(no of clusters)
    Plot SS (Silhouette score) Vs K(no of clusters)
    '''

    k_values = range(min_k, max_k+1)
    wcss_values = []
    ss_values = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k,random_state=0)
        kmeans.fit(X)
        label = kmeans.labels_
        centroids = kmeans.cluster_centers_

        sse_tot,krange,sse_list = wcss(X,label,centroids,k)
        wcss_values.append(sse_tot)

        ss_score = silhouette_score(X,label)
        ss_values.append(ss_score)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    ax[0].plot(k_values, wcss_values)
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('Sum of Squared Errors')

    ax[1].plot(k_values, ss_values)
    ax[1].set_xlabel('k')
    ax[1].set_ylabel('Silhouette Score')


if __name__ == '__main__':

    df = read_clean_data()
    #X = get_X(df)
    X = get_QX(df)

    #To plot WCSS & Silhoutte score to determine Optimum value for K
    #plot_wcss_silhouette_score(X/1000000,15, 25)
    #To plot all clusters
    #plot_clusters(df,'Cluster',X,no_clusters)
    #To plot single cluster
    #plot_cluster(df,'Cluster',X,clusterid=0)

    #Group1 : Housing prices <=500,000
    X1 = X[(X.ix[:,0] <= 500000) & (X.ix[:,0] <= 500000)]
    X1_pct = get_pct_change(X1)
    df_X1 = pd.merge(X1,df.ix[:,0:5],how='inner', left_index=True, right_on = ['Zipcode'])

    no_samples=1000
    no_clusters = 4
    df_X1 = assign_cluster(df_X1,X1,'Cluster',no_clusters)
    #plot_clusters(df_X1,'Cluster',X1,no_clusters)
    df_X1 = assign_cluster(df_X1,X1_pct,'ClusterPct',no_clusters)
    #plot_clusters(df_X1,'ClusterPct',X1,no_clusters)
    bc = BootstrapCluster(n_clusters=no_clusters,n_samples=no_samples)
    bc.fit(X1)
    df_X1['ClusterBS']=bc.labels
    #plot_clusters(df_X1,'ClusterBS',X1,no_clusters)
    bc.fit(X1_pct)
    df_X1['ClusterPctBS']=bc.labels
    #plot_clusters(df_X1,'ClusterPctBS',X1,no_clusters)

    #Check the clustering between ClusterBS and ClusterPCTBS
    #df_X1[['Zipcode','City','CountyName','ClusterBS','ClusterPctBS']][df_X1.ClusterBS==0]
    #df_X1[['Zipcode','City','CountyName','ClusterBS','ClusterPctBS']][df_X1.ClusterBS==1]
    #df_X1[['Zipcode','City','CountyName','ClusterBS','ClusterPctBS']][df_X1.ClusterBS==2]
    #df_X1[['Zipcode','City','CountyName','ClusterBS','ClusterPctBS']][df_X1.ClusterBS==3]

    ##Group2 : Housing prices > 500,000 and < 1,000,000
    X2 = X[(X.ix[:,0] > 500000) & (X.ix[:,0] <= 1000000)]
    X2_pct = get_pct_change(X2)
    df_X2 = pd.merge(X2,df.ix[:,0:5],how='inner', left_index=True, right_on = ['Zipcode'])

    no_samples=1000
    no_clusters = 5
    df_X2 = assign_cluster(df_X2,X2,'Cluster',no_clusters)
    bc = BootstrapCluster(n_clusters=no_clusters,n_samples=no_samples)
    bc.fit(X2)
    df_X2['ClusterBS']=bc.labels

    no_clusters = 4
    df_X2 = assign_cluster(df_X2,X2_pct,'ClusterPct',no_clusters)
    bc.fit(X2_pct)
    df_X2['ClusterPctBS']=bc.labels

    #Plotting Graphs
    #plot_clusters(df_X2,'Cluster',X2,no_clusters)
    #plot_clusters(df_X2,'ClusterBS',X2,no_clusters)
    #plot_clusters(df_X2,'ClusterPct',X2,no_clusters)
    #plot_clusters(df_X2,'ClusterPctBS',X2,no_clusters)

    ##Group3 : Housing prices > 1,000,000 and < 2,000,000
    X3 = X[(X.ix[:,0] > 1000000) & (X.ix[:,0] <= 2000000)]
    X3_pct = get_pct_change(X3)
    df_X3 = pd.merge(X3,df.ix[:,0:5],how='inner', left_index=True, right_on = ['Zipcode'])

    no_samples=1000
    no_clusters = 4
    df_X3 = assign_cluster(df_X3,X3,'Cluster',no_clusters)
    bc = BootstrapCluster(n_clusters=no_clusters,n_samples=no_samples)
    bc.fit(X3)
    df_X3['ClusterBS']=bc.labels

    df_X3 = assign_cluster(df_X3,X3_pct,'ClusterPct',no_clusters)
    bc.fit(X3_pct)
    df_X3['ClusterPctBS']=bc.labels

    #Plotting Graphs
    #plot_clusters(df_X3,'Cluster',X3,no_clusters)
    #plot_clusters(df_X3,'ClusterBS',X3,no_clusters)
    #plot_clusters(df_X3,'ClusterPct',X3,no_clusters)
    #plot_clusters(df_X3,'ClusterPctBS',X3,no_clusters)
