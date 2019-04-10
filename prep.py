import csv
import geohash2
import pprint
import time

def time_cnt(_time=None, time_itl = 60 * 60):
    if _time is None:
        return -1
    else:
        t = _time.tm_hour * 60 * 60 / time_itl
        t += _time.tm_min * 60 / time_itl
        t += _time.tm_sec / time_itl
        return int(t)


def idx_time(yday, time_cnt, num_day):
    return yday * num_day + time_cnt


def prepro_data(file_path, ret_cont = "req_serials", time_itl=60 * 60):
    file_15_1 = open(file_path, 'r')
    reader = csv.DictReader(file_15_1)

    orders = []
    num_itl = int(24 * 60 * 60 / time_itl)
    my_sta_set = set()
    pre_num = 25

    cnt_w = [0] * 7
    cnt_day = []
    y_day2w_day = {}
    max_time_idx = 0
    min_time_idx = int(1e9)

    for item in reader:
        if item['pickup_longitude'] != '0':
            _item0 = dict()
            _item0['S_geohash'] = geohash2.encode(float(item['pickup_longitude']), float(item['pickup_latitude']), 5)

            my_sta_set.add(_item0['S_geohash'])

            St = time.strptime(item['tpep_pickup_datetime'], '%Y-%m-%d %H:%M:%S')
            Tt = time.strptime(item['tpep_dropoff_datetime'], '%Y-%m-%d %H:%M:%S')

            idx_np = idx_time(St.tm_yday, time_cnt(St, time_itl), num_itl)
            y_day2w_day[St.tm_yday] = St.tm_wday
            min_time_idx = min(min_time_idx, idx_np)
            max_time_idx = max(max_time_idx, idx_np)

            _item0['St_week_day'] = St.tm_wday
            _item0['St_year_day'] = St.tm_yday
            _item0['S_time'] = time_cnt(St, time_itl)
            _item0['T_time'] = time_cnt(Tt, time_itl)
            _item0['St'] = St
            _item0['Tt'] = Tt
            _item0['dist'] = float(item['trip_distance'])

            orders.append(_item0)

    if ret_cont == "req_serials":
        req_serials = {item: [0 for i in range(max_time_idx - min_time_idx + 1)] for item in my_sta_set}
        for item in orders:
            day = item['St_year_day']
            time_day = item['S_time']
            _idx_time = idx_time(day, time_day, num_itl) - min_time_idx

            req_serials[item['S_geohash']][_idx_time] += 1

        return req_serials, my_sta_set
