import matplot.pyplot as plt

#更换坐标轴
new_xticks_positions = [0,1,2,3,4,5,6,7,8,9]
new_xticks_labels = ['1','2','3','4','5','6','7', '8','9','10']
plt.xticks(new_xticks_positions, new_xticks_labels)
plt.title('预测值与观测值比较 [单位：ms]')
