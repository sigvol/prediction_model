from pmdarima.arima import auto_arima
import numpy as np

def idx_pl2(x, n2):
    return x % n2


def idx_pl1(x, n1, n2):
    return int(x / n2) % n1


def reg(data=[]):
    sum_data = sum(data)

    if sum_data == 0:
        return data

    length = len(data)
    for i in range(length):
        data[i] /= sum_data / length

    return data


def get_prop_pl1(pl1_pl2=7, pl2_num=24, data=[]):
    prop_pl1 = [0] * pl1_pl2
    for i in range(len(data)):
        prop_pl1[idx_pl1(i, pl1_pl2, pl2_num)] += data[i]

    prop_pl1 = reg(prop_pl1)

    return prop_pl1


def get_prop_pl2(pl1_pl2=7, pl2_num=24, data=[]):
    prop_pl2 = [[0 for _ in range(pl2_num)] for _ in range(pl1_pl2)]
    for i in range(len(data)):
        prop_pl2[idx_pl1(i, pl1_pl2, pl2_num)][idx_pl2(i, pl2_num)] += data[i]

    for i in range(pl1_pl2):
        prop_pl2[i] = reg(prop_pl2[i])

    return prop_pl2


def pred_alg1(pl1_pl2=7, pl2_num=24, data=[], length=7 * 24):
    if length > pl1_pl2 * pl2_num:
        length = pl1_pl2 * pl2_num

    pl1_num = pl1_pl2 * pl2_num
    ave = sum(data[len(data) - pl1_num:]) / pl1_num
    prop_pl1 = get_prop_pl1(pl1_pl2, pl2_num, data)
    prop_pl2 = get_prop_pl2(pl1_pl2, pl2_num, data)

    preds = [0] * length
    data_length = len(data)

    for i in range(length):
        pl1 = idx_pl1(i + data_length, pl1_pl2, pl2_num)
        pl2 = idx_pl2(i + data_length, pl2_num)
        preds[i] = ave * prop_pl1[pl1] * prop_pl2[pl1][pl2]

    return preds


def pred_alg2(pl1_pl2=7, pl2_num=24, data=[], length=7 * 24, alpha=0.5, his_cnt=10):
    pl1_num = pl1_pl2 * pl2_num

    if length > pl1_num:
        length = pl1_num

    preds = [0] * length
    data_length = len(data)

    for i in range(length):
        p = alpha
        cnt = 0

        for j in range(his_cnt):
            hist_time = i + data_length - (j + 1) * pl1_num

            if hist_time < 0:
                break

            cnt += p
            preds[i] += data[hist_time] * p
            p *= (1 - alpha)

        preds[i] /= cnt

    return preds


def pred_alg3(pl1_pl2=7, pl2_num=24, data=[], length=7 * 24):
    pl1_num = pl1_pl2 * pl2_num

    modl = auto_arima(data, seasonal=True,
                      stepwise=True, suppress_warnings=True,
                      error_action='ignore')
    preds, conf_int = modl.predict(n_periods=length, return_conf_int=True)

    return preds


def pred_alg4(*preds, test, test_len=24 * 6, length=24, prop_alg='cpererr'):
    num_preds = len(preds)
    prop_pred = [0] * num_preds

    for i in range(num_preds):
        for j in range(test_len):
            if prop_alg == 'cpererr':
                prop_pred[i] += abs(test[j] - preds[i][j]) / (test[j] + preds[i][j] + 1) / test_len

        prop_pred[i] = 1 - prop_pred[i]

    pred_ensemble = [0] * length

    for i in range(length):
        for j in range(num_preds):
            pred_ensemble[i] += prop_pred[j] / sum(prop_pred) * preds[j][i + test_len]

    return pred_ensemble


def err_count(pred, test):
    test_len = len(test)

    smse = 0
    ae = 0

    for i in range(test_len):
        smse += (pred[i] - test[i]) ** 2
        ae += abs(pred[i] - test[i])

    smse = np.sqrt(smse / test_len)
    ae /= test_len

    return smse, ae
