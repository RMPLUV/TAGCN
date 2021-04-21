#计算每个网格的中心点位置

import pandas as pd
import numpy as np
from pandas import DataFrame
# df1 = pd.read_csv('E:\Projects\STGCN-PyTorch11\haikou_remove_time_exception.csv').to_numpy()
# df1 = df1[:,2:6]
# print(df1)
# exit()
path ="E:/Projects/STGCN-PyTorch5.0/data_process/"
df2 = pd.read_csv('E:\Projects\STGCN-PyTorch5.0\data\grid_scale4.0.csv').to_numpy()
df2 = df2[:, np.r_[0:5,7]]      #r_不连续
df2 = df2.astype(np.float64)
# df2 = df2[:5]
centrepoint = []

for (a,b,c,d,e,f) in df2:
    centrepoint.append(((c+e)/2,(d+f)/2))

centrepoint = np.array(centrepoint)
quyuid4 = np.concatenate((df2,centrepoint),axis=1)
# quyuid4 =np.hstack(df1,quyuid3)a
quyuid4 = list(quyuid4)
data =DataFrame(quyuid4)
print("ok!")
data.to_csv(path+str('make_centrepoint')+'.csv',
            header=['id','property','l_lng','u_lat','r_lng','d_lat','centre_lng','centre_lat'],index=False)
