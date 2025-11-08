#导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#配置中文与负号
plt.rcParams['font.sans-serif']=['simHei']
plt.rcParams['axes.unicode_minus']=False
#定义函数（能量密度=电压*电容/质量）
def test(a,b,c):
    y=a*b/c
    return y
#导入Excel文件，模拟大数据
df=pd.read_csv('材料数据.csv',index_col=0)
#用平均值填补漏洞
df=df.fillna(np.mean(df))
#提取为纯数组，方便计算
arr=df.values
#进行切片
arr_a=arr[:,1]
arr_b=arr[:,2]
arr_c=arr[:,3]
#调用函数，得到计算结果
result=[]
result=test(arr_a,arr_b,arr_c)
#生成x轴序号（与y轴同长）
x=np.arange(len(result))
#绘制图像
plt.plot(x,result,marker='^',color='red',label='材料能量密度')
plt.xlabel('材料序号')
plt.ylabel('能量密度')
plt.title('材料性能计算结果')
plt.legend()
plt.grid()
plt.show()
