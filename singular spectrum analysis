import numpy as np
#加载数据
data = np.loadtxt('',dtype=float) 

#时滞矩阵构建
def traj_matr(data, L, K):
    X = np.array([data[i:L + i] for i in range(0, K)]).T
    return X

#对角平均化
def average_anti_diag(X):
    X = np.asarray(X)
    out = [np.mean(X[::-1, :].diagonal(i)) for i in range(-X.shape[0] + 1, X.shape[1])]
    return np.asarray(out)

#设置窗口长度
windowLen = 365

X = traj_matr(LODR, windowLen, len(LODR) - windowLen + 1)
r = np.linalg.matrix_rank(X)
U, Sigma, VT = np.linalg.svd(X)
Xs = []

# 分组
for i in range(r):
    t = Sigma[i] * U[:, i][:, None] @ VT[i, :][:, None].T
    Xs.append(t)

# 反对角线均值，rec_martix即重构的分量
rec_martix = []
for i in range(r):
    rec_martix.append(average_anti_diag(Xs[i]))

# 计算贡献率
variance_explained = Sigma**2 / np.sum(Sigma**2)


