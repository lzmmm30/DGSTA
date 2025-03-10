import os
import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm
from libcity.data.dataset import TrafficStatePointDataset
from libcity.data.utils import generate_dataloader
from tslearn.clustering import TimeSeriesKMeans, KShape


class DGSTADataset(TrafficStatePointDataset):

    def __init__(self, config):
        self.type_short_path = config.get('type_short_path', 'hop')
        super().__init__(config)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'dgsta_point_based_{}.npz'.format(self.parameters_str))
        self.points_per_hour = 3600 // self.time_intervals
        self.dtw_matrix = self._get_dtw()
        self.points_per_day = 24 * 3600 // self.time_intervals
        self.cand_key_days = config.get("cand_key_days", 14)
        self.s_attn_size = config.get("s_attn_size", 3)
        self.n_cluster = config.get("n_cluster", 16)
        self.cluster_max_iter = config.get("cluster_max_iter", 5)
        self.cluster_method = config.get("cluster_method", "kshape")

    # 由所有原始数据生成DTW矩阵 dtw_matrix
    def _get_dtw(self):
        cache_path = './libcity/cache/dataset_cache/dtw_' + self.dataset + '.npy'
        for ind, filename in enumerate(self.data_files):
            if ind == 0:
                df = self._load_dyna(filename)  # 返回(17836,170,1)
            else:
                df = np.concatenate((df, self._load_dyna(filename)), axis=0)
        if not os.path.exists(cache_path):
            data_mean = np.mean(
                [df[24 * self.points_per_hour * i: 24 * self.points_per_hour * (i + 1)]
                 for i in range(df.shape[0] // (24 * self.points_per_hour))], axis=0)
            print("data_mean:" + data_mean)
            dtw_distance = np.zeros((self.num_nodes, self.num_nodes))
            for i in tqdm(range(self.num_nodes)):
                for j in range(i, self.num_nodes):
                    dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
            for i in range(self.num_nodes):
                for j in range(i):
                    dtw_distance[i][j] = dtw_distance[j][i]
            np.save(cache_path, dtw_distance)
        dtw_matrix = np.load(cache_path)
        # self._logger.info('Load DTW matrix from {}'.format(cache_path))
        return dtw_matrix

    # 加载rel文件，super()._load_rel()调用traffic_state_dataset的_load_rel()函数
    # 对邻接矩阵加工，得到节点之间最短距离
    def _load_rel(self):
        self.sd_mx = None
        super()._load_rel()
        self._logger.info('Max adj_mx value = {}'.format(self.adj_mx.max()))
        self.sh_mx = self.adj_mx.copy()  # copy邻接矩阵 (170,170)
        if self.type_short_path == 'hop':  # true
            self.sh_mx[self.sh_mx > 0] = 1
            self.sh_mx[self.sh_mx == 0] = 511  # 值为0的转为511
            for i in range(self.num_nodes):
                self.sh_mx[i, i] = 0  # 对角线元素置0
            # 弗洛伊德最短路径 (i,j)表示点i到点j的距离
            for k in range(self.num_nodes):
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        self.sh_mx[i, j] = min(self.sh_mx[i, j], self.sh_mx[i, k] + self.sh_mx[k, j], 511)
            np.save('{}.npy'.format(self.dataset), self.sh_mx)

    def _calculate_adjacency_matrix(self):
        self._logger.info("Start Calculate the weight by Gauss kernel!")
        self.sd_mx = self.adj_mx.copy()
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0
        if self.type_short_path == 'dist':
            self.sd_mx[self.adj_mx == 0] = np.inf
            for k in range(self.num_nodes):
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        self.sd_mx[i, j] = min(self.sd_mx[i, j], self.sd_mx[i, k] + self.sd_mx[k, j])

    # 将训练集、测试集、验证集归一化并转为dataloader
    # 加载kshape文件，变量pattern_keys
    # 返回训练集、测试集、验证集的dataloader
    def get_data(self):
        x_train, y_train, ind_train, x_val, y_val, ind_val, x_test, y_test, ind_test = [], [], [], [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, ind_train, x_val, y_val, ind_val, x_test, y_test, ind_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, ind_train, x_val, y_val, ind_val, x_test, y_test, ind_test = self._generate_train_val_test()
                # 训练集、测试集、验证集 (17836,12,170,9) np.array
        self.feature_dim = x_train.shape[-1]  # 9
        self.ext_dim = self.feature_dim - self.output_dim  # 9-1
        # scaler是一个类 有mean和std两个变量 求交通流的平均值和方差
        self.scaler = self._get_scalar(self.scaler_type,
                                       x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        # ext_scaler也是一个类
        self.ext_scaler = self._get_scalar(self.ext_scaler_type,
                                           x_train[..., self.output_dim:], y_train[..., self.output_dim:])
        # 归一化
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        if self.normal_external:  # false
            x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])
        train_data = list(zip(x_train, y_train, ind_train))
        eval_data = list(zip(x_val, y_val, ind_val))
        test_data = list(zip(x_test, y_test, ind_test))
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample,
                                distributed=self.distributed)
        # dataloader可以理解为将数据按batch分为一组一组的数据
        self.num_batches = len(self.train_dataloader)
        # 加载kshape文件
        self.pattern_key_file = os.path.join(
            './libcity/cache/dataset_cache/', 'pattern_keys_{}_{}_{}_{}_{}_{}'.format(
                self.cluster_method, self.dataset, self.cand_key_days, self.s_attn_size, self.n_cluster,
                self.cluster_max_iter))
        if not os.path.exists(self.pattern_key_file + '.npy'):
            cand_key_time_steps = self.cand_key_days * self.points_per_day
            pattern_cand_keys = x_train[:cand_key_time_steps, :self.s_attn_size, :, :self.output_dim].swapaxes(1,
                                                                                                               2).reshape(
                -1, self.s_attn_size, self.output_dim)
            self._logger.info("Clustering...")
            if self.cluster_method == "kshape":
                km = KShape(n_clusters=self.n_cluster, max_iter=self.cluster_max_iter).fit(pattern_cand_keys)
            else:
                km = TimeSeriesKMeans(n_clusters=self.n_cluster, metric="softdtw", max_iter=self.cluster_max_iter).fit(
                    pattern_cand_keys)
            self.pattern_keys = km.cluster_centers_
            np.save(self.pattern_key_file, self.pattern_keys)
            self._logger.info("Saved at file " + self.pattern_key_file + ".npy")
        # 走这里
        else:
            self.pattern_keys = np.load(self.pattern_key_file + ".npy")
            self._logger.info("Loaded file " + self.pattern_key_file + ".npy")
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "sd_mx": self.sd_mx, "sh_mx": self.sh_mx,
                "ext_dim": self.ext_dim, "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches,
                "dtw_matrix": self.dtw_matrix, "pattern_keys": self.pattern_keys}
    # scaler 平均值和方差
    # adj_mx 邻接矩阵 01
    # sd_mx None
    # sh_mx 节点之间最短距离
    # ext_dim 8
    # num_nodes 170
    # feature_dim 9
    # output_dim 1
    # num_batches 669
    # dtw_matrix (170,170)
    # pattern_keys (16,3,1)
