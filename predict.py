import csv
import geohash2
import pprint
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

file_15_1 = open("data/yellow_tripdata_2015-01.csv", 'r')
reader = csv.DictReader(file_15_1)

orders = []
time_itl = 60 * 60
num_itl = int(24 * 60 * 60 / time_itl)
my_sta_set = []
pre_num = 25


def time_cnt(_time=None):
    if _time is None:
        return -1
    else:
        t = _time.tm_hour * 60 * 60 / time_itl
        t += _time.tm_min * 60 / time_itl
        t += _time.tm_sec / time_itl
        return int(t)

cnt_w = [0] * 7
cnt_day = []
y_day2w_day = {}

for item in reader:
    # pprint.pprint(float(item['pickup_longitude']))
    if item['pickup_longitude'] != '0':
        _item0 = dict()
        _item0['S_geohash'] = geohash2.encode(float(item['pickup_longitude']), float(item['pickup_latitude']), 5)
        # _item0['T_geohash'] = geohash2.encode(float(item['dropoff_longitude']), float(item['dropoff_latitude']), 5)

        if (_item0['S_geohash'] in my_sta_set) is False:
            my_sta_set.append(_item0['S_geohash'])

        St = time.strptime(item['tpep_pickup_datetime'], '%Y-%m-%d %H:%M:%S')
        Tt = time.strptime(item['tpep_dropoff_datetime'], '%Y-%m-%d %H:%M:%S')

        if ((St.tm_yday in cnt_day) is False) and St.tm_yday < pre_num:
            cnt_w[St.tm_wday] += 1
            cnt_day.append(St.tm_yday)

        y_day2w_day[St.tm_yday] = St.tm_wday

        _item0['week_day'] = St.tm_wday
        # print(St.tm_wday)
        _item0['year_day'] = St.tm_yday
        _item0['S_time'] = time_cnt(St)
        _item0['T_time'] = time_cnt(Tt)

        _item0['dist'] = float(item['trip_distance'])
        # 距离为0合理嘛

        # pprint.pprint(dir(item))

        orders.append(_item0)

# pprint.pprint(len(orders))

order_cnt = {item: [[0 for i in range(num_itl)] for j in range(32)] for item in my_sta_set}
test_order_cnt = {item: [[0 for i in range(num_itl)] for j in range(7)] for item in
                  my_sta_set}  # 测试集中每个地点，周一至周日，每个时间片的需求数
pre_order_cnt = {item: [[0 for i in range(num_itl)] for j in range(7)] for item in
                 my_sta_set}  # 历史数据集中的每个地点，周一至周日，每个时间片的需求数
# pprint.pprint(my_sta_set)
tot_day = {item: [0 for i in range(num_itl)] for item in my_sta_set}  # 测试集前一周的平均需求总数
prop_week = {item: [0 for i in range(7)] for item in my_sta_set}  # 每一个地点一周中每一天所占比例
prop_time_period = {item: [[0 for i in range(num_itl)] for j in range(7)] for item in my_sta_set}
# 每一个地点周一至周日每一个时间片所占比例

print(len(pre_order_cnt))

pprint.pprint(my_sta_set[10])
exit(0)

for item in orders:
    yday = item['year_day']
    S_sta = item['S_geohash']
    wday = item['week_day']
    S_time = item['S_time']

    order_cnt[S_sta][yday][S_time] += 1.0
    if (yday in range(pre_num - 7, pre_num)) is True:  # 统计前一周的平均需求
        tot_day[S_sta][S_time] += 1.0 / 7.0
    if yday < pre_num:
        pre_order_cnt[S_sta][wday][S_time] += 1.0 / cnt_w[wday]
        # pprint.pprint(pre_order_cnt[S_sta])
        prop_week[S_sta][wday] += 1.0
        prop_time_period[S_sta][wday][S_time] += 1.0 / cnt_w[wday]
    else:
        test_order_cnt[S_sta][wday][S_time] += 1.0

for sta in my_sta_set:  # 统计比例项
    sum_week = sum(prop_week[sta])
    for i in range(7):
        prop_week[sta][i] /= sum_week / 7.0
        sum_day = sum(prop_time_period[sta][i])
        for j in range(num_itl):
            if prop_time_period[sta][i][j] != 0:
                prop_time_period[sta][i][j] /= sum_day / num_itl

pre = [{}, {}, {}, {}, {}]
pre[4] = {item: [[0 for i in range(num_itl)] for j in range(7)] for item in my_sta_set}
pre[3] = {item: [[0 for i in range(num_itl)] for j in range(7)] for item in my_sta_set}
pre[2] = {item: [[0 for i in range(num_itl)] for j in range(7)] for item in my_sta_set}
pre[1] = {item: [[0 for i in range(num_itl)] for j in range(7)] for item in my_sta_set}

pre_ave_error = [0] * 5
per_error = [{item: 0 for item in my_sta_set} for i in range(4)]
all_status = len(my_sta_set) * num_itl

alpha = 0.5  # 方法二平滑系数
his_p = 3

for sta in my_sta_set:
    for i in range(7):  # 使用方法一进行预测
        for j in range(num_itl):
            pre[1][sta][i][j] = tot_day[sta][j] * prop_week[sta][i] * prop_time_period[sta][i][j]

            n_y_day = pre_num + i
            n_w_day = y_day2w_day[n_y_day]
            count = 1  # 使用方法二进行预测
            for k in range(his_p):
                if k < his_p - 1:
                    p = (1 - alpha) ** k * alpha
                    pre[2][sta][n_w_day][j] += p * order_cnt[sta][n_y_day - 7 * (k + 1)][j]
                    count -= p
                else:
                    pre[2][sta][n_w_day][j] += count * order_cnt[sta][n_y_day - 7 * (k + 1)][j]

            if n_y_day < 31:
                per_error[1][sta] += abs(pre[1][sta][i][j] - test_order_cnt[sta][i][j]) / \
                                        (pre[1][sta][i][j] + test_order_cnt[sta][i][j] + 1) / (6.0 * num_itl)
                per_error[2][sta] += abs(pre[2][sta][n_w_day][j] - test_order_cnt[sta][n_w_day][j]) / \
                                        (pre[2][sta][n_w_day][j] + test_order_cnt[sta][n_w_day][j] + 1) / \
                                        (6.0 * num_itl)
            else:
                pre_ave_error[1] += abs(pre[1][sta][i][j] - test_order_cnt[sta][i][j]) ** 2 / all_status
                pre_ave_error[2] += abs(pre[2][sta][n_w_day][j] - test_order_cnt[sta][n_w_day][j]) ** 2 / all_status

for sta in my_sta_set:  # 使用方法三进行预测
    train = []
    test = []
    for i in range(pre_num):
        train += order_cnt[sta][i]
    for i in range(pre_num, 32):
        test += order_cnt[sta][i]

    # plt.plot(range(len(train)), train, range(len(train), len(train) + len(test)), test)
    # break

    modl = auto_arima(train, seasonal=True, m=num_itl,
                      stepwise=True, suppress_warnings=True,
                      error_action='ignore')
    preds, conf_int = modl.predict(n_periods=len(test), return_conf_int=True)
    pre_ave_error[3] += sum((test[6 * num_itl:] - preds[6 * num_itl:]) ** 2) / num_itl / len(my_sta_set)
    # print("Test RMSE: %.3f" % np.sqrt(sum((test - preds) ** 2) / len(test)))

    for i in range(7):
        pre[3][sta][y_day2w_day[pre_num + i]] = preds[i * num_itl: (i + 1) * num_itl]

    per_test = test[: 6 * num_itl]
    per_pre = preds[: 6 * num_itl]
    per_error[3][sta] += sum((per_test - per_pre) / (per_test + per_pre + 1)) / (6.0 * num_itl)
    if (sta is my_sta_set[10]) is True:
        plt.plot(range(len(test)), preds, range(len(test)), test)


# pprint.pprint(cnt_w)
# pprint.pprint(tot_day[my_sta_set[10]])
# pprint.pprint(test_order_cnt[my_sta_set[10]][2])
# pprint.pprint(prop_week[sta])
# pprint.pprint(prop_time_period[my_sta_set[10]][3])

for sta in my_sta_set:  # 使用方法四进行预测
    cnt_per_error = 0
    for i in range(1, 4):
        cnt_per_error += 1 - per_error[i][sta]
    for i in range(1, 4):
        for j in range(num_itl):
            pre[4][sta][y_day2w_day[31]][j] += pre[i][sta][y_day2w_day[31]][j] * (1 - per_error[i][sta]) / cnt_per_error
    for j in range(num_itl):
        pre_ave_error[4] += (pre[4][sta][y_day2w_day[31]][j] - test_order_cnt[sta][y_day2w_day[31]][j]) ** 2 / \
                                len(my_sta_set) / num_itl

'''
test_sta = 1
test_method = 2

print(pre_ave_error[test_method])
# pprint.pprint(my_sta_set)
fig, axs = plt.subplots(2, 4)
for i in range(4):
    axs[0, i].plot(range(num_itl), pre[test_method][my_sta_set[test_sta]][i],
                   range(num_itl), test_order_cnt[my_sta_set[test_sta]][i])
    if i < 3:
        axs[1, i].plot(range(num_itl), pre[test_method][my_sta_set[test_sta]][i + 4],
                       range(num_itl), test_order_cnt[my_sta_set[test_sta]][i + 4])
plt.show()
'''
test_sta = 10
fig, axs = plt.subplots(2, 2)

for i in range(1, 5):
    pre_ave_error[i] = np.sqrt(pre_ave_error[i])
    pprint.pprint("error %d:" % i)
    pprint.pprint(pre_ave_error[i])
    if i < 4:
        pprint.pprint("prop %d in test_station:" % i)
        pprint.pprint(1 - per_error[i][my_sta_set[test_sta]])


axs[0, 0].plot(range(num_itl), pre[1][my_sta_set[test_sta]][y_day2w_day[31]],
               range(num_itl), test_order_cnt[my_sta_set[test_sta]][y_day2w_day[31]])
axs[0, 1].plot(range(num_itl), pre[2][my_sta_set[test_sta]][y_day2w_day[31]],
               range(num_itl), test_order_cnt[my_sta_set[test_sta]][y_day2w_day[31]])
axs[1, 0].plot(range(num_itl), pre[3][my_sta_set[test_sta]][y_day2w_day[31]],
               range(num_itl), test_order_cnt[my_sta_set[test_sta]][y_day2w_day[31]])
axs[1, 1].plot(range(num_itl), pre[4][my_sta_set[test_sta]][y_day2w_day[31]],
               range(num_itl), test_order_cnt[my_sta_set[test_sta]][y_day2w_day[31]])

plt.show()
# pprint.pprint(pre1[my_sta_set[10]][2])
# pprint.pprint(test_order_cnt[my_sta_set[10]][2])
# pprint.pprint(test_order_cnt[my_sta_set[0]][2])
# pprint.pprint(test_order_cnt[my_sta_set[1]][2])
