# encoding utf-8
'''
@Author: william
@Description:  制作节点权重
@time:2020/6/15 19:21
'''

import pandas as pd
from datetime import datetime, timedelta
import calendar
from math import radians, cos, sin, asin, sqrt
import numpy as np


def Cal_V_matrix(data):
    day_minute_list = pd.date_range(start='5/1/2017', end='7/31/2017', freq='120T').format(formatter=lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    # V_matrix_list = []
    result = []
    for day_minute in day_minute_list:     #day_minute_list 从5.1号到6.5号的0点到24点共6913个时间片
        day_minute_start = datetime.strptime(day_minute, '%Y-%m-%d %H:%M:%S')
        if datetime.strptime(day_minute_start.strftime('%H:%M:%S'), '%H:%M:%S') < datetime.strptime('08:00:00', '%H:%M:%S'):
            continue
        if datetime.strptime(day_minute_start.strftime('%H:%M:%S'), '%H:%M:%S') > datetime.strptime('22:00:00', '%H:%M:%S'):
            continue
        # 终止时间只要6点--22点之间的。舍弃6点之前，22点之后的
        day_minute_end = day_minute_start + timedelta(minutes=120)    #时间片结束时间06：30，07：00……21：30，22：00
        print(str(day_minute_start) + ' - ' + str(day_minute_end))
        temp_result = []
        time_count = 0

        # temp_data_1筛选出  时间片起始时间(包括6点之前的)  <=  出发时间  <  到达时间  <=  时间片结束时间(6点之后)  的数据
        temp_data_1 = data[(data.departure_time >= day_minute_start.strftime('%Y-%m-%d %H:%M:%S')) & (
                data.arrive_time <= day_minute_end.strftime('%Y-%m-%d %H:%M:%S'))]
        # temp_data_2筛选出  时间片起始时间(包括6点之前的)  <=  出发时间  <=  时间片结束时间  <  到达时间
        temp_data_2 = data[(data.departure_time >= day_minute_start.strftime('%Y-%m-%d %H:%M:%S')) & (
                data.departure_time <= day_minute_end.strftime('%Y-%m-%d %H:%M:%S')) & (
                                       data.arrive_time > day_minute_end.strftime('%Y-%m-%d %H:%M:%S'))]
        # temp_data_2筛选出  出发时间  <  时间片起始时间  <=  到达时间  <=  时间片结束时间
        temp_data_3 = data[(data.arrive_time >= day_minute_start.strftime('%Y-%m-%d %H:%M:%S')) & (
                data.arrive_time <= day_minute_end.strftime('%Y-%m-%d %H:%M:%S')) & (
                                       data.departure_time < day_minute_start.strftime('%Y-%m-%d %H:%M:%S'))]
        # 按行拼接。将5/1 09：05之前的数据按3种情况分好
        temp_data = pd.concat([temp_data_1, temp_data_2, temp_data_3], axis=0, ignore_index=True)
        df1 = temp_data.values[:, np.r_[6, 8]]
        df1 = df1.astype(np.float64)
        if len(temp_data) == 0:
            continue
        for i in range(1,25):
            for j in range(1, 25):
                count = 0
                for (a, b) in df1:           #按切分好时间的数据进行遍历
                    # if ((a == j) & (b == i)) | ((a == i) & (b == j)) &(i != j):
                    if ((a == j) & (b == i)) & (i != j):
                        count += 1
                temp_result.append(count)
        result.append(temp_result)
        print()
    V_matrix = np.array(result)
    return V_matrix


if __name__ == '__main__':
    now_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    data = pd.read_csv(r'E:\Projects\STGCN-PyTorch5.0\data_process\pointtogrid.csv')

    V_matrix = Cal_V_matrix(data)
    np.savetxt("V_matrix_" + now_time + ".csv", V_matrix, delimiter=',', fmt='%f')
    print()
    #V_matrix_2020_10_23_19_47_36