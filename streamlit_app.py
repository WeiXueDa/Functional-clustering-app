import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from sklearn.cluster import KMeans
import scipy
import os
from io import BytesIO, StringIO
import zipfile

# 强制使用CPU，禁用CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device('cpu')

# 设置页面配置
st.set_page_config(page_title="功能性聚类分析工具", layout="wide")

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 全局变量存储分析结果
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'k_range' not in st.session_state:
    st.session_state.k_range = (2, 6)
if 'current_k' not in st.session_state:
    st.session_state.current_k = 2

# 辅助函数：将matplotlib图表转换为PDF并返回BytesIO对象
def fig_to_pdf_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight', dpi=300)
    buf.seek(0)
    return buf

# 核心算法代码（保持原有功能）
class FunClu:
    def __init__(self, K=3, seed=None,
                 mean_type='power_equation', covariance_type="SAD1_tied",
                 lr=1e-2, tol=1e-3, reg_covar=1e-6, max_iter=100,
                 init_params="kmeans"):
        torch.set_default_dtype(torch.float64)
        self.elbo = None
        self.parameters = {}
        self.eps = np.finfo(float).eps
        self.is_fit = False
        self.reg_covar = reg_covar

        # 强制使用CPU
        self.device = torch.device('cpu')
        
        # 模型方法属性
        self.f_mean = self.power_equation
        self.f_mean_init = self.power_equation
        self.f_mean_closed_form = False

        self.f_covariance = self.get_SAD1_tied
        self.f_cov_closed_form = False

        self.mean_type = mean_type
        self.covariance_type = covariance_type

        self.reg_covar = reg_covar
        self.init_params = init_params

        # 仅保留D模型
        self.model_type = 'D'

        if seed:
            self.seed = seed
        else:
            self.seed = random.randint(0, 100000)

        self.hyperparameters = {
            "K": K,
            "seed": self.seed,
            "learning_rate": lr,
            "optimizer": 'Adam',
            "bounds_l_mu": None,
            "bounds_u_mu": None,
            "bounds_l_sig": None,
            "bounds_u_sig": None,
            "mean_type": mean_type,
            "covariance_type": covariance_type,
            "mean_function": self.f_mean,
            "covariance_function": self.f_covariance,
            "mean_init_function": self.f_mean_init,
            "covariance_init_function": None,
            "number_mean_pars": 2,
            "number_covariance_pars": 2
        }
    # 幂函数模型

    def power_equation(self, x, *pars):
        a, b = pars[0], pars[1]
        # 使用默认双精度（与第一版一致），不强制指定float32
        x = torch.as_tensor(x, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        b = torch.as_tensor(b, device=self.device)
        y = a * x ** b
        return y

    # SAD1协方差结构
    def get_SAD1_tied(self, pars, d):
        # 恢复参数含义：pars[0]=phi（自相关系数），pars[1]=gamma（缩放因子）
        phi, gamma = pars[0], pars[1]
        # 计算符合SAD1理论的对角线元素
        range_tensor = torch.tensor(range(1, d + 1), dtype=torch.float32, device=self.device)
        diag = (1 - phi ** (2 * range_tensor)) / (1 - phi ** 2 + 1e-8)  # 加小值避免除零
        # 构建Toeplitz矩阵（相关性衰减）
        toeplitz_matrix = torch.tensor(
         scipy.linalg.toeplitz(range(0, d), range(0, d)), 
         device=self.device,  # 使用默认双精度（float64）
         dtype=torch.float64 
         )
        # 计算协方差矩阵并缩放
        SIGMA = diag * (phi ** toeplitz_matrix)
        SIGMA = gamma ** 2 * (SIGMA.tril() + SIGMA.T.triu() - torch.diag(diag))  # 确保对称
        return SIGMA

    def _pre_process_data(self):
        # 关键修改：同时处理NaN和0值（将0视为缺失值）
        inds = np.where(np.isnan(self.X) | (self.X == 0))
        X_copy = self.X.copy()
        # 用每列非缺失（非NaN且非0）的均值填充
        col_means = np.nanmean(np.where(self.X == 0, np.nan, self.X), axis=0)
        X_copy[inds] = np.take(col_means, inds[1])
        return X_copy

    def _initialize(self, X, times=None, trans_data=False):
        if (isinstance(X, pd.DataFrame)):
            self.data_colnames = list(X)
            self.data_rownames = list(X.index)
        else:
            self.data_colnames = None
            self.data_rownames = None

        try:
            times[0]
            self.times = times
            self.X = np.where(np.array(X) == 0, np.nan, np.array(X))
        except Exception as e:
            # 保持原始times计算方式（不添加额外微小值）
            X_nan_zero = np.where(np.array(X) == 0, np.nan, np.array(X))
            times = np.log10(np.nansum(X_nan_zero, axis=0))
            self.order = np.argsort(times)
            times_new = times[self.order]

            self.X = np.where(np.array(X) == 0, np.nan, np.array(X))[:, self.order]
            self.times = times_new

        if trans_data:
            self.X = np.log10(self.X + 1)
        else:
            pass

        self.N, self.D = self.X.shape
        K = self.hyperparameters["K"]

        # 分割数据
        self.N0, self.X_comp, self.N1, self.X_incomp, self.n0, self.n1, self.N2, self.n2, self.X_empty = self.split_data(
            self.X)

        if (self.N1 == 0):
            self.contain_missing = False
            X_filled = self.X.copy()
        else:
            self.contain_missing = True
            X_filled = self._pre_process_data()

        if self.init_params == "kmeans":
            resp = np.zeros((self.N, K))
            label = (KMeans(n_clusters=K, n_init='auto', random_state=self.seed).fit(X_filled).labels_)
            resp[np.arange(self.N), label] = 1
        elif self.init_params == "random":
            np.random.seed(seed=self.seed)
            resp = np.random.uniform(size=(self.N, K))
            resp /= resp.sum(axis=0)[:, np.newaxis]

        # 权重、中心和协方差矩阵
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X_filled) / nk[:, np.newaxis]
        weight = (nk / self.N)

        self.parameters = {
            "weight": torch.tensor(weight, device=self.device),
            "resp": torch.tensor(resp, device=self.device),
            "means": torch.tensor(means, device=self.device),
            "covariances": None,
            "resp0": None,
            "resp1": None
        }

        if self.mean_type != 'None':
            pars_means_init = list(map(lambda k:
                                       self.get_mean_init(x=times,
                                                          y=k,
                                                          mean_func=self.f_mean_init,
                                                          n_mean_pars=self.hyperparameters['number_mean_pars']),
                                       means))

            pars_means_init = torch.tensor(np.array(pars_means_init), requires_grad=True, device=self.device)
            self.parameters["pars_means"] = pars_means_init
            self.parameters["means"] = list(map(lambda k: self.f_mean(self.times, *k), pars_means_init))
            self.parameters["means"] = torch.vstack(self.parameters["means"]).detach()
        else:
            self.parameters["pars_means"] = torch.tensor(means, device=self.device)
            self.hyperparameters["number_mean_pars"] = K * self.D

        if self.f_cov_closed_form == True:
            self.parameters["pars_covariance"] = self._estimate_cov_parameters(X_filled, resp, self.reg_covar,
                                                                              self.covariance_type)
            self.parameters["covariances"] = self._estimate_cov_parameters(X_filled, resp, self.reg_covar,
                                                                          self.covariance_type)
            self.hyperparameters['number_covariance_pars'] = self._n_cov_parameters(K, self.D, self.covariance_type)
        else:
            if 'full' in self.f_covariance.__name__:
                self.parameters["pars_covariance"] = torch.tensor(np.random.random(K * 2)).reshape(K, 2).to(self.device)
                self.parameters["pars_covariance"] = self.parameters["pars_covariance"].requires_grad_(True)
                self.parameters["covariances"] = self.f_covariance(self.parameters["pars_covariance"], self.D)
                self.hyperparameters['number_covariance_pars'] = self.parameters["pars_covariance"].size(0) * 2
            else:
                self.parameters["pars_covariance"] = torch.tensor(np.array([0.1, 0.1]), requires_grad=True, device=self.device)
                self.parameters["covariances"] = self.f_covariance(self.parameters["pars_covariance"], self.D).repeat(K,
                                                                                                                      1,
                                                                                                                      1)

        # 转换为tensor并确保在CPU上
       # 转换为tensor时显式指定float64，与第一版完全一致
        self.X = torch.tensor(self.X, requires_grad=False, device=self.device, dtype=torch.float64)
        self.times = torch.tensor(self.times, requires_grad=False, device=self.device, dtype=torch.float64)
        self.X_comp = torch.tensor(self.X_comp, requires_grad=False, device=self.device, dtype=torch.float64)
        self.X_incomp = torch.tensor(self.X_incomp, requires_grad=False, device=self.device, dtype=torch.float64)
        if (self.N2 > 0):
            self.X_empty = torch.tensor(self.X_empty, requires_grad=False, device=self.device, dtype=torch.float64)

        self.splited_X, self.observed_position = self.get_observed_position(self.X_incomp)

        if self.contain_missing == True:
            self.missed_position = list(
                map(lambda n: torch.tensor(np.setdiff1d(list(range(self.D)), n.cpu().numpy()), dtype=int, device=self.device),
                    self.observed_position))

            self.position_rearranged = torch.stack(list(map(lambda x, y: torch.cat((x, y)), self.observed_position,
                                                            self.missed_position)))
            self.new_position = torch.stack(list(map(torch.argsort, self.position_rearranged)))
        else:
            self.missed_position, self.position_rearranged, self.new_position = None, None, None

    def split_data(self, X):
        N, D = X.shape
        tmp = np.argwhere(np.isnan(X))

        if any(np.unique(tmp[:, 0], return_counts=True)[1] == D):
            n2 = np.unique(tmp[:, 0])[np.where(np.unique(tmp[:, 0], return_counts=True)[1] == D)[0]]
            N2 = len(n2)
            X_empty = X[n2,]

            n1 = np.unique(tmp[:, 0])[np.where(np.unique(tmp[:, 0], return_counts=True)[1] < D)[0]]
            N1 = len(n1)
            X_incomp = X[n1, :]

            n0 = np.setdiff1d(np.unique(np.argwhere(X)[:, 0]), np.union1d(n1, n2))
            N0 = len(n0)
            X_comp = X[n0, :]
        else:
            n2 = None
            N2 = 0
            X_empty = None

            n1 = np.unique(tmp[:, 0])[np.where(np.unique(tmp[:, 0], return_counts=True)[1] < D)[0]]
            N1 = len(n1)
            X_incomp = X[n1, :]

            n0 = np.setdiff1d(np.unique(np.argwhere(X)[:, 0]), n1)
            N0 = len(n0)
            X_comp = X[n0, :]
        return [N0, X_comp, N1, X_incomp, n0, n1, N2, n2, X_empty]

    def get_mean_init(self, x, y, mean_func, n_mean_pars, maxit=1e5):
        # 将输入转换为NumPy数组（确保不在GPU上）
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
            
        params_0 = np.ones(n_mean_pars)
        par_est = scipy.optimize.curve_fit(mean_func, xdata=x, ydata=y, p0=params_0, maxfev=int(maxit))[0]
        return par_est

    def _n_cov_parameters(self, n_components, n_features, covariance_type):
        if covariance_type == "full":
            cov_params = n_components * n_features * (n_features + 1) / 2.0
        elif covariance_type == "diag":
            cov_params = n_components * n_features
        elif covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif covariance_type == "spherical":
            cov_params = n_components
        return int(cov_params)

    def _estimate_log_gaussian_prob(self, X, means, covariances):
        K = means.shape[0]
        if (len(covariances.shape) != 3):
            covariances = covariances.repeat(K, 1, 1)
        mvn_log = list(
            map(lambda x, y: torch.distributions.MultivariateNormal(x, y).log_prob(X), means, covariances))
        return torch.stack(mvn_log)

    def _estimate_gaussian_covariances_full(self, resp, X, nk, means, reg_covar):
        n_components, n_features = means.shape
        covariances = np.empty((n_components, n_features, n_features))
        for k in range(n_components):
            diff = X - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
            covariances[k].flat[:: n_features + 1] += reg_covar
        return covariances

    def _estimate_gaussian_covariances_tied(self, resp, X, nk, means, reg_covar):
        avg_X2 = np.dot(X.T, X)
        avg_means2 = np.dot(nk * means.T, means)
        covariance = avg_X2 - avg_means2
        covariance /= nk.sum()
        covariance.flat[:: len(covariance) + 1] += reg_covar
        return covariance

    def _estimate_gaussian_covariances_diag(self, resp, X, nk, means, reg_covar):
        avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
        avg_means2 = means ** 2
        avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
        return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

    def _estimate_gaussian_covariances_spherical(self, resp, X, nk, means, reg_covar):
        return self._estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)

    def _estimate_cov_parameters(self, X, resp, reg_covar, covariance_type):
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        covariances = {
            "full": self._estimate_gaussian_covariances_full,
            "tied": self._estimate_gaussian_covariances_tied,
            "diag": self._estimate_gaussian_covariances_diag,
            "spherical": self._estimate_gaussian_covariances_spherical,
        }[covariance_type](resp, X, nk, means, reg_covar)
        return torch.tensor(covariances, device=self.device)

    def _calculate_incomp_responsibility(self, X_comp, splited_X, observed_position, weight, times,
                                         pars_means, pars_covariance,
                                         f_mean, f_covariance):
        N0, D = X_comp.shape
        N1 = len(splited_X)
        K = len(weight)

        means = torch.stack(list(map(lambda k: f_mean(times, *k), pars_means)))
        covariances = f_covariance(pars_covariance, D)
        if len(covariances.shape) != 3:
            covariances = covariances.repeat(K, 1, 1)

        mvn_log0 = self._estimate_log_gaussian_prob(X_comp, means, covariances)
        mvn0 = mvn_log0.T + torch.log(weight).expand(N0, K)
        log_resp0 = mvn0 - torch.logsumexp(mvn0, 1).expand(K, N0).T

        mvn_log1 = self._eval_incomp_special(splited_X, observed_position, means, covariances)
        mvn1 = torch.stack(mvn_log1) + torch.log(weight).expand(N1, K)
        log_resp1 = mvn1 - torch.logsumexp(mvn1, 1).expand(K, N1).T

        return [log_resp0, log_resp1]

    def loss_function(self, X_comp, splited_X, observed_position, times, pars_means, pars_covariance,
                      resp0, resp1, f_mean, f_covariance, weight, eps=2e-30):
        N0, D = X_comp.shape
        N1 = len(splited_X)
        K = len(weight)

        means = torch.stack(list(map(lambda k: f_mean(times, *k), pars_means)))
        covariances = f_covariance(pars_covariance, D)
        if len(covariances.shape) != 3:
            covariances = f_covariance(pars_covariance, D).repeat(K, 1, 1)

        mvn_log0 = list(map(lambda x, y: self.MVN_normal_full(X_comp, x, y), means, covariances))
        mvn0 = torch.stack(mvn_log0).T + torch.log(weight).expand(N0, K)
        LLB0 = -torch.sum(resp0 * mvn0 - resp0 * torch.log(resp0 + eps))

        mvn_log1 = self._eval_incomp_special(splited_X, observed_position, means, covariances)
        mvn1 = torch.stack(mvn_log1) + torch.log(weight).expand(N1, K)
        LLB1 = -torch.sum(resp1 * mvn1 - resp1 * torch.log(resp1 + eps))
        LLB = LLB0 + LLB1

        return LLB

    def loss_maximization(self, n_epochs, optimizer, X_comp, splited_X, observed_position, times,
                          pars_means, pars_covariance,
                          resp0, resp1,
                          f_mean, f_covariance,
                          weight, eps):
        for epoch in range(1, n_epochs + 1):
            LLB = self.loss_function(X_comp, splited_X, observed_position, times, pars_means, pars_covariance,
                                     resp0, resp1, f_mean, f_covariance, weight, eps)
            optimizer.zero_grad()
            LLB.backward()
            optimizer.step()
            if epoch % 100 == 0:
                st.write(f'Epoch {epoch}, LLB {float(LLB)}')

        return [pars_means, pars_covariance]

    def MVN_normal_full(self, X, mean, covariance):
        mvn = torch.distributions.MultivariateNormal(mean, covariance).log_prob(X)
        return mvn

    def _eval_incomp_special(self, splited_X, observed_position, means, covariances):
        N = len(splited_X)
        all_means = means.repeat(N, 1, 1)
        all_means_observed = list(map(lambda x, y: x[:, y], all_means, observed_position))

        all_covariances = covariances.repeat(N, 1, 1, 1)

        def _split_covariance(all_covariance, observed_position):
            res = torch.stack(list(map(lambda x: x[observed_position][:, observed_position], all_covariance)))
            return res

        all_covariances_observed = list(map(_split_covariance, all_covariances, observed_position))
        mvn = list(map(self._estimate_log_gaussian_prob, splited_X, all_means_observed, all_covariances_observed))
        return mvn

    def get_observed_position(self, X):
        tmp = torch.argwhere(~torch.isnan(X))
        number_observed = tmp[:, 0].unique(return_counts=True)[1]
        observed_position = tmp[:, 1].split(number_observed.tolist())
        splited_X = list(map(lambda x, y: x[y], X, observed_position))
        return [splited_X, observed_position]

    def _fill_means(self, X, mean, covariance, observed_position, missed_position, resp):
        N, D = len(X), len(mean)
        all_mean = mean.repeat(N, 1)
        all_mean_observed = list(map(lambda x, y: x[y], all_mean, observed_position))

        all_covariances = covariance.repeat(N, 1, 1)

        all_covariance_observed = list(map(lambda x, y: x[y][:, y], all_covariances, observed_position))
        all_covariance_missed = list(map(lambda x, y: x[y][:, y], all_covariances, missed_position))
        all_covariance_res = list(
            map(lambda x, y, z: x[y][:, z], all_covariances, observed_position, missed_position))

        all_covariance_observed_inv = list(map(torch.linalg.inv, all_covariance_observed))

        all_mean_missed = list(map(lambda x, y: x[y], all_mean, missed_position))
        res = list(map(lambda x, mean, sigma: torch.mm((x - mean).unsqueeze(0), sigma),
                       X, all_mean_observed, all_covariance_observed_inv))
        all_missed_conditional = list(
            map(lambda x, y, z: x + torch.mm(y, z), all_mean_missed, res, all_covariance_res))

        all_mean_expectation = torch.stack(list(map(lambda x, y: torch.cat((x, y.squeeze(0))),
                                                    X, all_missed_conditional)))
        missed_number = np.array(list(map(len, X)))
        missed_value_expectation = list(map(lambda x, y, z: x[y:] * z, all_mean_expectation, missed_number, resp))

        return missed_value_expectation

    def _likelihood_lower_bound(self):
        P, K = self.parameters, self.hyperparameters["K"]
        weight, resp, resp0, resp1, means, covariances = P["weight"], P["resp"], P["resp0"], P["resp1"], P["means"], P[
            "covariances"]
        pars_means, pars_covariance = P["pars_means"], P["pars_covariance"]

        if (self.contain_missing == True):
            likelihood_lower_bound = self.loss_function(self.X_comp, self.splited_X, self.observed_position,
                                                       self.times, pars_means, pars_covariance,
                                                       resp0, resp1,
                                                       self.f_mean, self.f_covariance,
                                                       weight, self.eps)
        else:
            means = torch.stack(list(map(lambda k: self.f_mean(self.times, *k), pars_means)))
            covariances = self.f_covariance(pars_covariance, self.D)
            mvn_log = self._estimate_log_gaussian_prob(self.X, means, covariances)
            mvn = mvn_log.T + torch.log(weight).expand(self.N, K)
            likelihood_lower_bound = -torch.sum(resp * mvn - resp * torch.log(resp + self.eps))

        return likelihood_lower_bound

    def _E_step(self):
        P, K = self.parameters, self.hyperparameters["K"]
        weight, resp0, resp1, means, covariances = P["weight"], P["resp0"], P["resp1"], P["means"], P["covariances"]
        pars_means, pars_covariance = P["pars_means"], P["pars_covariance"]

        if (self.contain_missing == True):
            log_resp0, log_resp1 = self._calculate_incomp_responsibility(self.X_comp,
                                                                        self.splited_X,
                                                                        self.observed_position,
                                                                        weight,
                                                                        self.times,
                                                                        pars_means,
                                                                        pars_covariance,
                                                                        self.f_mean,
                                                                        self.f_covariance)
            P["resp0"] = torch.exp(log_resp0).detach()
            P["resp1"] = torch.exp(log_resp1).detach()
            P["resp"] = torch.vstack([P["resp0"], P["resp1"]])
        else:
            means = torch.stack(list(map(lambda k: self.f_mean(self.times, *k), pars_means)))
            covariances = self.f_covariance(pars_covariance, self.D)

            mvn_log = self._estimate_log_gaussian_prob(self.X, means, covariances)
            mvn = mvn_log.T + torch.log(weight).expand(self.N, K)
            log_resp = mvn - torch.logsumexp(mvn, 1).expand(K, self.N).T
            P["resp"] = torch.exp(log_resp).detach()

    def _M_step(self, n_epochs=50):
        P, K = self.parameters, self.hyperparameters["K"]
        resp, resp0, resp1 = P["resp"], P["resp0"], P["resp1"]
        weight, pars_means, pars_covariance = P["weight"], P["pars_means"], P["pars_covariance"]

        # 更新聚类先验
        nk = torch.sum(resp, axis=0)
        P["weight"] = (nk / self.N).detach() + torch.tensor(self.eps, device=self.device)

        if self.contain_missing:
            optimizer = torch.optim.Adam([pars_means, pars_covariance],
                                         lr=self.hyperparameters["learning_rate"])
            pars_means, pars_covariance = self.loss_maximization(n_epochs,
                                                               optimizer,
                                                               self.X_comp,
                                                               self.splited_X,
                                                               self.observed_position,
                                                               self.times,
                                                               pars_means,
                                                               pars_covariance,
                                                               resp0,
                                                               resp1,
                                                               self.f_mean,
                                                               self.f_covariance,
                                                               weight,
                                                               eps=self.eps)
        else:
            optimizer = torch.optim.Adam([pars_means, pars_covariance],
                                         lr=self.hyperparameters["learning_rate"])
            for epoch in range(1, n_epochs + 1):
                means = torch.stack(list(map(lambda k: self.f_mean(self.times, *k), pars_means)))
                covariances = self.f_covariance(pars_covariance, self.D)

                mvn_log = self._estimate_log_gaussian_prob(self.X, means, covariances)
                mvn = mvn_log.T + torch.log(weight).expand(self.N, K)
                LLB = -torch.sum(resp * mvn - resp * torch.log(resp + self.eps))
                optimizer.zero_grad()
                LLB.backward()
                optimizer.step()

        P["pars_means"], P["pars_covariance"] = pars_means, pars_covariance

    def _n_parameters(self):
        K, n_features = self.parameters['pars_means'].shape
        cov_params = self.hyperparameters['number_covariance_pars']

        if self.covariance_type == "full":
            cov_params = K * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = K * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = K

        mean_params = self.hyperparameters['number_mean_pars'] * K

        if mean_params == 0:
            mean_params = n_features * K

        return int(cov_params + mean_params + K - 1)

    def fit(self, X, times=None, trans_data=False, max_iter=50, tol=1e-3, verbose=True, verbose_interval=1):
        prev_vlb = -np.inf
        self._initialize(X, times, trans_data)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for _iter in range(max_iter):
            try:
                self._E_step()
                self._M_step()

                vlb = self._likelihood_lower_bound().detach()

                if verbose:
                    if _iter % verbose_interval == 0:
                        status_text.text(f"迭代 {_iter + 1}/{max_iter}. 对数似然下界 = {vlb:.4f}")

                progress = (_iter + 1) / max_iter
                progress_bar.progress(progress)

                converged = _iter > 0 and torch.abs(vlb - prev_vlb) <= tol
                if torch.isnan(vlb) or converged:
                    break

                prev_vlb = vlb

            except torch.linalg.LinAlgError:
                st.error("奇异矩阵：组件崩溃")
                return -1

        # 更新参数
        means = torch.stack(list(map(lambda k: self.f_mean(self.times, *k), self.parameters['pars_means'])))
        self.parameters["means"] = means

        covariances = self.f_covariance(self.parameters["pars_covariance"], self.D)
        if (len(covariances.shape) != 3):
            covariances = covariances.repeat(self.hyperparameters["K"], 1, 1)
        self.parameters["covariances"] = covariances

        # 更新超参数
        self.elbo = vlb
        self.is_fit = True
        self.hyperparameters['BIC'] = self.bic()
        self.hyperparameters['max_iter'] = max_iter
        self.hyperparameters['tol'] = tol

        return 0

    def bic(self):
        return 2 * self._likelihood_lower_bound().detach().cpu().numpy() + self._n_parameters() * np.log(self.N * self.D)

    def _fill_missing(self):
        if self.contain_missing:
            resp1 = self.parameters["resp1"]
            filled_means = list(map(lambda x, y, z: self._fill_means(
                self.splited_X, x, y, self.observed_position, self.missed_position, z),
                self.parameters["means"], self.parameters["covariances"], resp1.T))
            filled_means2 = list(map(list, zip(*filled_means)))
            filled_means2 = list(map(sum, filled_means2))

            X_filled = self.X_incomp.clone()
            for i in range(self.N1):
                X_filled[i][self.missed_position[i]] = filled_means2[i]
            self.X_filled = X_filled
        else:
            self.X_filled = self.X_comp.clone()

    def get_results(self):
        """返回聚类结果数据，用于可视化"""
        self._fill_missing()
        
        pars_means = self.parameters['pars_means'].detach().cpu().numpy()
        label = np.argmax(self.parameters["resp"].cpu().numpy(), 1) + 1
        label = label.astype(int)
        
        if self.N1 == 0:  # 无缺失数据
            X = self.X_comp.detach().cpu().numpy()
            X_filled = X
            
            new_ID = np.array(self.data_rownames)[self.n0.astype(int)]
            label2 = label
        elif self.N2 > 0:  # 有全缺失样本
            X = np.vstack([
                self.X_comp.detach().cpu().numpy(),
                self.X_incomp.detach().cpu().numpy(),
                self.X_empty.detach().cpu().numpy()
            ])
            X_filled = np.vstack([
                self.X_comp.detach().cpu().numpy(),
                self.X_filled.detach().cpu().numpy(),
                self.X_empty.detach().cpu().numpy()
            ])
            
            new_ID = np.concatenate([
                np.array(self.data_rownames)[self.n0.astype(int)],
                np.array(self.data_rownames)[self.n1.astype(int)],
                np.array(self.data_rownames)[self.n2.astype(int)]
            ])
            label2 = np.concatenate([label, np.repeat(-1, self.N2)])
        else:  # 有缺失但无全缺失样本
            X = np.vstack([
                self.X_comp.detach().cpu().numpy(),
                self.X_incomp.detach().cpu().numpy()
            ])
            X_filled = np.vstack([
                self.X_comp.detach().cpu().numpy(),
                self.X_filled.detach().cpu().numpy()
            ])
            
            new_ID = np.concatenate([
                np.array(self.data_rownames)[self.n0.astype(int)],
                np.array(self.data_rownames)[self.n1.astype(int)]
            ])
            label2 = label
        
        # 确保标签与数据长度匹配
        if len(label2) != X.shape[0]:
            if len(label2) > X.shape[0]:
                label2 = label2[:X.shape[0]]
            else:
                label2 = np.pad(label2, (0, X.shape[0]-len(label2)), constant_values=-1)
        
        times = self.times.detach().cpu().numpy()
        
        # 创建结果字典
        results = {
            'X': X,
            'X_filled': X_filled,
            'times': times,
            'labels': label2,
            'ids': new_ID,
            'pars_means': pars_means,
            'pars_covariance': self.parameters['pars_covariance'].detach().cpu().numpy(),
            'hyperparameters': self.hyperparameters,
            'bic': self.bic()
        }
        
        return results

# 主应用
def main():
    st.title("功能性聚类分析工具")
    
     # 创建自定义样式的标签页
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;  /* 增加标签之间的间隔 */
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.25rem;  /* 增大标签字体大小 */
            padding: 10px 20px;  /* 增加标签内边距 */
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["上传数据", "聚类分析", "BIC可视化", "聚类结果可视化"])
    
    with tab1:
        st.header("上传数据文件")
        uploaded_file = st.file_uploader("选择CSV文件", type=["csv"])
        
        if uploaded_file is not None:
            # 读取数据，假设第一列为行名，第一行为列名
            df = pd.read_csv(uploaded_file, index_col=0)
            st.session_state.data = df
            
            # 显示数据基本信息
            st.subheader("数据预览")
            st.dataframe(df.head())
            
            st.subheader("数据信息")
            st.write(f"样本数: {df.shape[0]}")
            st.write(f"特征数: {df.shape[1]}")
            
            # 关键修改：同时统计NaN和0值作为缺失值
            original_missing = df.isnull().sum().sum()  # 原始NaN数量
            zero_count = (df == 0).sum().sum()          # 0值数量
            total_missing = original_missing + zero_count  # 总缺失值（NaN+0）
            
            # 显示细分的缺失值统计
            st.write(f"**缺失值总数**: {total_missing}")  # 突出显示总缺失值
            
            # 警告信息更新为总缺失值
            if total_missing > 0:
                st.warning(f"数据中包含{total_missing}个缺失数据（含{original_missing}个NaN和{zero_count}个0值），将在分析过程中进行处理")
    
    with tab2:
        st.header("聚类分析参数设置")
        
        if st.session_state.data is None:
            st.info("请先在第一个标签页上传数据")
        else:
            # 设置K值范围
            col1, col2 = st.columns(2)
            with col1:
                min_k = st.number_input("最小K值", min_value=2, value=2)
            with col2:
                max_k = st.number_input("最大K值", min_value=min_k+1, value=11)
            
            st.session_state.k_range = (min_k, max_k)
            # 新增：种子设置输入框
            seed = st.number_input(
                "随机种子", 
                min_value=0, 
                value=2025,  # 默认保留2025，可手动修改
                help="设置随机种子以保证结果可复现，相同种子会得到相同聚类结果"
        )
            # 其他参数
            max_iter = st.slider("最大迭代次数", 10, 100, 30)
            lr = st.number_input("学习率", value=0.01, format="%.6f")
            
            if st.button("开始聚类分析"):
                st.session_state.results = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 对每个K值进行聚类
                for i, k in enumerate(range(min_k, max_k+1)):
                    status_text.text(f"正在分析K={k}...")
                    
                    # 初始化并拟合模型
                    model = FunClu(K=k, seed=seed, mean_type='power_equation', covariance_type="SAD1_tied", lr=lr)
                    model.fit(st.session_state.data, trans_data=True, max_iter=max_iter)
                    
                    # 获取结果
                    results = model.get_results()
                    results['cluster_weights'] = model.parameters["weight"].detach().cpu().numpy()
                    st.session_state.results[k] = results
                    
                    # 更新进度
                    progress_bar.progress((i+1)/(max_k - min_k + 1))
                
                status_text.text("聚类分析完成!")
                st.success(f"已完成K={min_k}到K={max_k}的聚类分析")
                st.info("请在第三个标签页查看BIC可视化结果")
    
    with tab3:
        st.header("BIC值可视化")
        
        if not st.session_state.results:
            st.info("请先在第二个标签页完成聚类分析")
        else:
            # 提取BIC值
            k_values = sorted(st.session_state.results.keys())
            bic_values = [st.session_state.results[k]['bic'] for k in k_values]
            
            # 找到BIC最小的K值
            best_k = k_values[np.argmin(bic_values)]
            
            # 绘制BIC折线图
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_values, bic_values, 'o-', color='blue')
            ax.axvline(x=best_k, color='red', linestyle='--', label=f'最佳K={best_k}')
            ax.set_xlabel('K值 (聚类数量)')
            ax.set_ylabel('BIC值')
            ax.set_title('不同K值的BIC值')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # 添加BIC图下载按钮
            st.subheader("下载BIC可视化结果")
            bic_pdf = fig_to_pdf_bytes(fig)
            st.download_button(
                label="下载BIC折线图 (PDF格式)",
                data=bic_pdf,
                file_name=f'bic_plot_k{min(k_values)}-{max(k_values)}.pdf',
                mime='application/pdf'
            )
            
            st.subheader("下载结果文件")
            selected_k = st.selectbox("选择K值", k_values)
                
            # 准备下载文件
            if st.button(f"下载K={selected_k}的结果文件"):
                results = st.session_state.results[selected_k]
                
                # 创建一个ZIP文件
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # 1. df_after.csv
                    df_after = pd.DataFrame(results['X'], index=results['ids'], columns=results['times'])
                    df_after['cluster'] = results['labels']
                    csv_data = df_after.to_csv(index=True)
                    zipf.writestr(f'XD_pheno_merged_K_{selected_k}_df_after.csv', csv_data)
                    
                    # 2. filled_after.csv
                    df_filled = pd.DataFrame(results['X_filled'], index=results['ids'], columns=results['times'])
                    df_filled['cluster'] = results['labels']
                    csv_data = df_filled.to_csv(index=True)
                    zipf.writestr(f'XD_pheno_merged_K_{selected_k}_filled_after.csv', csv_data)
                    
                    # 3. pars_means.csv
                    np.savetxt('temp.csv', results['pars_means'], delimiter=',')
                    with open('temp.csv', 'r') as f:
                        zipf.writestr(f'XD_pheno_merged_K_{selected_k}_pars_means.csv', f.read())
                    # 新增：4. 聚类中心数据点文件（每个聚类中心的具体数据）
                    # 计算每个聚类中心的实际数据点（使用幂函数参数生成）
                    cluster_centers = []
                    times = results['times']  # 获取时间点/特征点

                    # 转换时间点为合适的列名格式（去除科学计数法，保留合适小数位数）
                    # 根据时间点数据特点调整格式化方式
                    formatted_times = [f"{t:.17f}".rstrip('0').rstrip('.') if '.' in f"{t:.17f}" else f"{t:.0f}" for t in times]

                    for pars in results['pars_means']:
                        # 使用幂函数计算每个时间点的中心值
                        center_values = pars[0] * times ** pars[1]
                        cluster_centers.append(center_values)

                    # 转换为DataFrame并保存，列名为具体时间点
                    centers_df = pd.DataFrame(
                        cluster_centers,
                        columns=formatted_times,  # 使用格式化后的时间点作为列名
                        index=[f'Cluster_{i+1}' for i in range(len(cluster_centers))]
                    )
                    zipf.writestr(f'XD_pheno_merged_K_{selected_k}_cluster_centers.csv', centers_df.to_csv(index=True))
                    
                    # 5. pars_covariance.csv
                    np.savetxt('temp.csv', results['pars_covariance'], delimiter=',')
                    with open('temp.csv', 'r') as f:
                        zipf.writestr(f'XD_pheno_merged_K_{selected_k}_pars_covariance.csv', f.read())

                    # 6. info_file.csv
                    info_data = results['hyperparameters'].copy()
                    cluster_weights = results['cluster_weights']  # 从results获取权重
                    log_cluster_weights = np.log(cluster_weights)
        
                    info_data['num_clusters'] = selected_k
                    info_data['cluster_weights_sum'] = np.sum(cluster_weights)
        
                    for i in range(selected_k):
                        info_data[f'cluster_{i+1}_weight'] = cluster_weights[i]
                        info_data[f'cluster_{i+1}_log_weight'] = log_cluster_weights[i]
        
                    info_df = pd.DataFrame.from_dict(info_data, orient='index', columns=['value'])
                    csv_data = info_df.to_csv(header=True)
                    zipf.writestr(f'XD_pheno_merged_K_{selected_k}_info_file.csv', csv_data)

                
                # 提供下载
                st.download_button(
                    label=f"下载K={selected_k}的ZIP文件",
                    data=zip_buffer.getvalue(),
                    file_name=f'XD_pheno_merged_K_{selected_k}.zip',
                    mime='application/zip'
                )
    
    with tab4:
        st.header("聚类结果可视化")
        
        if not st.session_state.results:
            st.info("请先在第二个标签页完成聚类分析")
        else:
            # 选择K值
            k_values = sorted(st.session_state.results.keys())
            selected_k = st.selectbox("选择K值查看聚类结果", k_values)
            
            # 获取所选K值的结果
            results = st.session_state.results[selected_k]
            n_clusters = selected_k
            
            # 绘制聚类结果，固定3列布局
            st.subheader(f'K={selected_k}的聚类结果')
            
            # 计算需要的行数（每行固定3列）
            max_per_row = 3
            rows = (n_clusters + max_per_row - 1) // max_per_row  # 向上取整
            
            # 为每个聚类准备数据
            cluster_data = []
            for cluster_id in range(1, n_clusters + 1):
                cluster_mask = results['labels'] == cluster_id
                cluster_samples = results['X_filled'][cluster_mask]
                cluster_times = results['times']
                pars = results['pars_means'][cluster_id - 1]
                cluster_data.append((cluster_id, cluster_samples, cluster_times, pars, np.sum(cluster_mask)))
            
            # 创建整体画布（固定3列）
            fig, axs = plt.subplots(rows, max_per_row, figsize=(5 * max_per_row, 4 * rows))
            fig.suptitle(f'K={selected_k}的聚类结果', fontsize=16)
            
            # 定义刻度格式化函数
            def format_tick(x, pos):
                return f'10^{x:.2f}'
            
            # 统一处理子图数组格式
            if rows == 1:
                axs = np.array([axs])
            if max_per_row == 1:
                axs = axs.reshape(-1, 1)
            
            # 为每个聚类绘制子图
            for i, (cluster_id, samples, times, pars, count) in enumerate(cluster_data):
                row_idx = i // max_per_row
                col_idx = i % max_per_row
                ax = axs[row_idx, col_idx]
                
                # 绘制散点图
                for sample in samples:
                    ax.scatter(times, sample, alpha=0.5, s=10)
                
                # 绘制幂函数曲线
                x_range = np.linspace(min(times), max(times), 100)
                y_fit = pars[0] * x_range ** pars[1]
                ax.plot(x_range, y_fit, 'r-', linewidth=2, label=f'幂函数: y = {pars[0]:.2f}x^{pars[1]:.2f}')
                
                # 设置坐标轴格式
                ax.xaxis.set_major_formatter(plt.FuncFormatter(format_tick))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_tick))
                
                ax.set_xlabel('Habitat Index')
                ax.set_ylabel('Niche Index')
                ax.set_title(f'M {cluster_id} (n={count})')
                ax.legend(fontsize=8)
                ax.grid(True)
            
            # 移除多余的子图（如果有的话）
            for i in range(n_clusters, rows * max_per_row):
                row_idx = i // max_per_row
                col_idx = i % max_per_row
                fig.delaxes(axs[row_idx, col_idx])
            
            # 调整布局，为总标题留出空间
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # 显示整体图表
            st.pyplot(fig)
            
            # 添加聚类结果图下载按钮
            st.subheader("下载聚类结果图")
            # 将图表转换为PDF
            buf = BytesIO()
            fig.savefig(buf, format='pdf', bbox_inches='tight', dpi=300)
            buf.seek(0)
            
            # 提供下载按钮
            st.download_button(
                label=f"下载K={selected_k}的聚类结果图 (PDF格式)",
                data=buf,
                file_name=f'cluster_results_k{selected_k}.pdf',
                mime='application/pdf'
            )

        
if __name__ == "__main__":
    main()
