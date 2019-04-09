import numpy as np
import matplotlib.pyplot as plt


def main():
    metrics = ["AUC_judd", "sAUC", "NSS", "CC", "SIM"]
    rate = {}
    mean = [{}, {}]


    data = load_files("./SalEMA30_metrics.npy", "./SalGANmid_metrics.npy")
    for i, x in enumerate(metrics):


        mean[0][x] = np.nanmean(data[0][:,i])
        mean[1][x] = np.nanmean(data[1][:,i])

        data_substracted = np.array (np.nan_to_num(data[0][:,i]) - np.nan_to_num(data[1][:,i]))
        #np.nan_to_num(data_substracted)
        #print(data_substracted)



        x_positive = np.where( data_substracted >= 0 )
        x_negative = np.where( data_substracted < 0 )

        if x == "NSS":
            print("SalEMA is better by a margin higher than 4 at {}".format(np.where(data_substracted > 1.5)))
            print("SalCLSTM is better by a margin higher than 4 at {}".format(np.where(data_substracted < -1.5)))

        y_positive = data_substracted[x_positive]
        y_negative = data_substracted[x_negative]

        rate[x] = len(y_positive)/(len(data_substracted))

        plot_scatter (x_positive, y_positive, "SalEMA  avg:" + str(float("{0:.4f}".format(mean[0][x]))) , x_negative, y_negative, "SalCLSTM  avg:"+ str(float("{0:.4f}".format(mean[1][x]))), "frames" + "  rate: " + str(float("{0:.2f}".format(rate[x]*100))) + " %", x)
        plt.savefig('{}.png'.format(metrics[i]))

    print("SalEMA: " + str(mean[0]))
    print("SalCLSTM: " + str(mean[1]))
    print("# of samples where SalEMA is doing better than SalCLSTM:  " + str(rate))
    #plt.show()

def plot_scatter (x1, y1, name1, x2, y2, name2, Xaxis_name, Yaxis_name):

    fig, ax = plt.subplots(figsize=(13,8))
    ax.scatter(x1, y1,  label=name1)
    ax.scatter(x2, y2, label=name2)

    legend = ax.legend(loc='lower right', shadow=False, fontsize='x-large')
    plt.grid(True)

    plt.xlabel(Xaxis_name)
    plt.ylabel(Yaxis_name)



def load_files(path1, path2):
    data1 = np.asarray( np.load(path1) )
    data2 = np.asarray( np.load(path2) )

    return [data1, data2]

if __name__ == '__main__':
    main()
