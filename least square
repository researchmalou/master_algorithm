import numpy as np

#加载数据
main_term = np.load('')
forecast_step = 5
data_model = main_term[:-forecast_step]
data_test = main_term[-forecast_step:]
time = np.arange(1, len(data_model) + 1)

# 构造设计矩阵
# 存放时间、常数项、线性趋势项
design_matrix = np.column_stack((
    np.ones(len(time)),
    time,
    time*time,
    np.sin(2 * np.pi * time / 365.24),
    np.cos(2 * np.pi * time / 365.24),
    np.sin(2 * np.pi * time / 182.62),
    np.cos(2 * np.pi * time / 182.62),
))
# 计算参数
params, residuals, _, _ = np.linalg.lstsq(design_matrix, data_model, rcond=None)
fit = np.dot(design_matrix, params)

# 构造未来设计矩阵
future_time = np.arange(len(data_model) + 1, len(data_model) + forecast_step + 1)
future_design_matrix = np.column_stack((
    np.ones(len(future_time)),
    future_time,
    future_time*future_time,
    np.sin(2 * np.pi * future_time / 365.24),
    np.cos(2 * np.pi * future_time / 365.24),
    np.sin(2 * np.pi * future_time / 182.62),
    np.cos(2 * np.pi * future_time / 182.62),
))

# 进行外推得到预测值
forecast = np.dot(future_design_matrix, params)
