# -*- coding: utf-8 -*-
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
       #                     random_state=0)

#X = StandardScaler().fit_transform(X)
spheres = np.load('/home/nader/scratch/lobsterpts_3d_new.npy')
xmin=-520
xmax = -460
ymin = 60
ymax = 140
xlim = spheres[np.where(np.logical_and(np.greater_equal(xmax,spheres[:,0]),np.less_equal(xmin, spheres[:,0])))]
lims = xlim[np.where(np.logical_and(np.greater_equal(ymax, xlim[:,1]),np.less_equal(ymin, xlim[:,1])))]
X = lims[:,0:3]
#X = spheres[:,0:3]
print(np.shape(X))

# #############################################################################
# Compute DBSCAN
test = 'n'
if test=='y':
    for i in np.arange(0.1,2,0.1):
        db = DBSCAN(eps=i, min_samples=2).fit(X)
        #db = DBSCAN(min_samples=1).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('eps = {0}'.format(i))
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        ##print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        #print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        #print("Adjusted Rand Index: %0.3f"
        #      % metrics.adjusted_rand_score(labels_true, labels))
        #print("Adjusted Mutual Information: %0.3f"
        #      % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f\n\n"
             % metrics.silhouette_score(X, labels))

epss, min_sampless = 0.7, 3
db = DBSCAN(epss, min_sampless).fit(X)
#db = DBSCAN(min_samples=1).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
##print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f\n\n"
     % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

cluster_meds = []
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=14)
    if len(xy)>0:
        cl = np.mean(xy, axis=0)
        cluster_meds.append(cl)
        plt.scatter(cl[0], cl[1])
    xy = X[class_member_mask & ~core_samples_mask]
    # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# plt.scatter(cluster_meds[:][0], cluster_meds[:][1])
np.savetxt('/home/nader/scratch/cluster_centroids_{0}_{1}_mean.csv'.format(epss, min_sampless),cluster_meds)