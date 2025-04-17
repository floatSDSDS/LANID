from tqdm import tqdm
import copy
from collections import defaultdict
import numpy as np
import torch

from utils.tools import trange


class MyDBSCAN():
    def __init__(self, eps, minpts):
        self.eps = eps
        self.minpts = minpts
        self.data = None

        self.clusters_all = None
        self.clusters_pts = defaultdict(None)  # cluster_cores[-1] includes outliers
        self.cluster_cores = defaultdict(None)

        self.flag_init = True
        self.c_eps_change = 0.1

    def fit(self, input_data):
        # print('> dbscan fitting...')
        self.counter = 0
        self.data_len = len(input_data)
        self.data = [{'idx': i, 'label': 0, 'nei': [], 'n_nei': 0, 'type': 0, 'val': val} for
                     i, val in enumerate(copy.deepcopy(list(torch.tensor(input_data))))]
        self.data_matrix = copy.deepcopy(torch.tensor(input_data))
        # distance_matrix = torch.matmul(self.dbscan.data_matrix, self.dbscan.data_matrix.T)

        if self.flag_init:
            self.initialize(input_data)

        current_label = 1
        for idx in trange(len(self.data), desc='dbscan'):
            pt = self.data[idx]
        # for idx, pt in enumerate(self.data):
            # print('processing: ' + str(idx) + ' ' + str(pt))
            if pt['label'] != 0:
                continue

            neighbor_loc = self.regionQuery(pt, current_label)

            if len(neighbor_loc) >= self.minpts:
                self.data[idx]['label'] = current_label
                self.counter += 1
                self.growCluster(neighbor_loc, current_label)
            else:
                self.data[idx]['label'] = -1
                self.data[idx]['type'] = -1
            self.counter = 0
            current_label += 1
        self.summary_cluster()

    def summary_cluster(self):
        self.clusters_pts = defaultdict(None)  # cluster_cores[-1] includes outliers
        self.cluster_cores = defaultdict(None)
        self.clusters_all = np.unique(np.array([d['label'] for d in self.data]))

        for c in self.clusters_all:
            self.clusters_pts[c] = [d['idx'] for d in self.data if d['label'] == c]
            self.cluster_cores[c] = [idx for idx in self.clusters_pts[c] if self.data[idx]['type'] == 1]

        core_all = [cl for cl in self.data if cl['type'] == 1]
        if -1 in self.clusters_pts.keys():
            print(f'> dbscan: {len(self.clusters_pts[-1])} outliers, '
                  f'{len(core_all)} cores in {len(self.clusters_all)-1} clusters.')
        else:
            print(f'> dbscan: no outlier, {len(self.clusters_all)} clusters.')

    def regionQuery(self, chosen_pt, current_label):
        idx = chosen_pt['idx']
        distance = torch.sum(torch.sqrt((chosen_pt['val'] - self.data_matrix) ** 2), dim=1)
        neighbor_loc = list(torch.where(distance <= self.eps)[0])

        self.data[idx]['nei'] = neighbor_loc
        self.data[idx]['n_nei'] = len(neighbor_loc)
        if len(neighbor_loc) >= self.minpts:
            self.data[idx]['type'] = 1
        return neighbor_loc

    def growCluster(self, neighbor_locs, current_label):
        i = 0
        while (i < len(neighbor_locs)):
            idx = neighbor_locs[i]
            if self.data[idx]['label'] == -1:
                self.data[idx]['label'] = current_label
                self.counter += 1
            elif self.data[idx]['label'] == 0:
                self.data[idx]['label'] = current_label
                self.counter += 1
                temp_neighbor_locs = self.regionQuery(self.data[idx], current_label)
                if len(temp_neighbor_locs) >= self.minpts:
                    neighbor_locs = neighbor_locs + temp_neighbor_locs
            i += 1

    def update_param(self, eps=None, minpts=None):
        if eps:
            self.eps = eps
            print(f'> update dbscan eps to {eps}.')
        if minpts:
            self.minpts = minpts
            print(f'> update dbscan minpts to {minpts}.')

    def initialize(self, input_data):

        self.flag_init = False

        dist_all = torch.cdist(self.data_matrix, self.data_matrix, 1)
        dist_topk = torch.topk(dist_all, k=2, largest=False)[0][:, 1]  # nearest distance
        self.update_param(eps=float(dist_topk.mean()))

        self.fit(input_data)

        if -1 in self.clusters_pts.keys():
            if len(self.clusters_pts[-1]) == len(self.data):
                print(f'all outliers, set minpts=1, '
                      f'increase search eps as {self.c_eps_change} * {self.eps}.')
                self.update_param(eps=self.eps*(1+self.c_eps_change), minpts=1)
                self.flag_init = True
        else:
            # only one cluster
            if len(self.clusters_pts.keys()) == 1:
                print(f'only one cluster, set minpts={self.minpts} - 1.')
                self.update_param(minpts=self.minpts-1)
                self.flag_init = True


