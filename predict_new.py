import predalg
import prep
import matplotlib.pyplot as plt
import pprint

time_itl = 15 * 60
num_itl = int(24 * 60 * 60 / time_itl)

time_serials, my_sta_set = prep.prepro_data(file_path="data/yellow_tripdata_2015-01.csv",
                                            ret_cont="req_serials", time_itl=time_itl)

for sta in my_sta_set:
    test_ed = len(time_serials[sta]) - num_itl * 7
    pred1 = predalg.pred_alg1(pl1_pl2=7, pl2_num=num_itl, data=time_serials[sta][:test_ed], length=7 * num_itl)
    pred2 = predalg.pred_alg2(pl1_pl2=7, pl2_num=num_itl, data=time_serials[sta][:test_ed], length=7 * num_itl)
    pred3 = predalg.pred_alg3(pl1_pl2=7, pl2_num=num_itl, data=time_serials[sta][:test_ed], length=7 * num_itl)
    pred4 = predalg.pred_alg4(pred1, pred2, test=time_serials[sta][test_ed: test_ed + 6 * num_itl],
                              test_len=6 * num_itl, length=num_itl)

    if sum(time_serials[sta][test_ed + 6 * num_itl:]) > 200:
        pprint.pprint(sta)
        pprint.pprint("alg1: SMSE %f/ AE %f" % predalg.err_count(pred1[6 * num_itl:],
                                                                 time_serials[sta][test_ed + 6 * num_itl:]))
        pprint.pprint("alg2: SMSE %f/ AE %f" % predalg.err_count(pred2[6 * num_itl:],
                                                                 time_serials[sta][test_ed + 6 * num_itl:]))
        pprint.pprint("alg3: SMSE %f/ AE %f" % predalg.err_count(pred3[6 * num_itl:],
                                                                 time_serials[sta][test_ed + 6 * num_itl:]))

        pprint.pprint("alg4: SMSE %f/ AE %f" % predalg.err_count(pred4,
                                                                 time_serials[sta][test_ed + 6 * num_itl:]))

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(range(num_itl), pred1[6 * num_itl:], range(num_itl), time_serials[sta][test_ed + 6 * num_itl:])
        axs[0, 1].plot(range(num_itl), pred2[6 * num_itl:], range(num_itl), time_serials[sta][test_ed + 6 * num_itl:])
        axs[1, 0].plot(range(num_itl), pred3[6 * num_itl:], range(num_itl), time_serials[sta][test_ed + 6 * num_itl:])
        axs[1, 1].plot(range(num_itl), pred4, range(num_itl), time_serials[sta][test_ed + 6 * num_itl:])

        plt.show()
    '''
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(range(7 * num_itl), pred1, range(7 * num_itl), time_serials[sta][test_ed:])
    axs[0, 1].plot(range(7 * num_itl), pred2, range(7 * num_itl), time_serials[sta][test_ed:])
    axs[1, 0].plot(range(7 * num_itl), pred3, range(7 * num_itl), time_serials[sta][test_ed:])
    '''

    # plt.show()


