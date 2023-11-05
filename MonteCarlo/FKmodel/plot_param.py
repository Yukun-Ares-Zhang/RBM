import torch
import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/Library/Fonts/Songti.ttc')
 
weight = torch.load(".\Trained_weight_SGD.pth")
#定义热图的横纵坐标
xLabel = ['1','2','3','4','5','6','7','8']
yLabel = ['1','2','3','4','5','6','7','8']

# for k in range(10):
#     for l in range(10):

#准备数据阶段，利用random生成二维数据（5*5）
data = []
for i in range(8):
    temp = []
    for j in range(8):
        temp.append(weight[0,8*i+j].item())
    data.append(temp)

#作图阶段
fig = plt.figure()
#定义画布为1*1个划分，并在第1个位置上进行作图
ax = fig.add_subplot(111)
#定义横纵坐标的刻度
ax.set_yticks(range(len(yLabel)))
ax.set_yticklabels(yLabel)
ax.set_xticks(range(len(xLabel)))
ax.set_xticklabels(xLabel)
#作图并选择热图的颜色填充风格，这里选择hot
im = ax.imshow(data, cmap=plt.cm.hot_r)
#增加右侧的颜色刻度条
plt.colorbar(im)
#增加标题
plt.suptitle("weight[0]_SGD")
#show
plt.savefig(".\weightPLOT\weight_SGD.png")
print(data)

