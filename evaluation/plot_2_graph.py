import numpy as np
import matplotlib.pyplot as plt


def main():
    metrics = ["AUC_judd", "sAUC", "NSS", "CC", "SIM"]
    rate = {}
    mean = [{}, {}]


    data = load_files("./SalEMA30_metrics.npy", "./SGmid_metrics.npy")
    for i, x in enumerate(metrics):


        mean[0][x] = np.nanmean(data[0][:,i])
        mean[1][x] = np.nanmean(data[1][:,i])

        data_substracted = np.array (np.nan_to_num(data[0][:,i]) - np.nan_to_num(data[1][:,i]))
        #np.nan_to_num(data_substracted)
        #print(data_substracted)

        x_positive = np.where( data_substracted >= 0 )
        x_negative = np.where( data_substracted < 0 )

        y_positive = data_substracted[x_positive]
        y_negative = data_substracted[x_negative]

        rate[x] = len(y_positive)/(len(data_substracted))

        plot_scatter (x_positive, y_positive, "SalEMA advantage", x_negative, y_negative, "SalCLSTM advantage", "Validation set videos", "{} difference".format(x))
        plt.savefig('{}.png'.format(metrics[i]))

    print("SalEMA: " + str(mean[0]))
    print("SalCLSTM: " + str(mean[1]))
    print("# of samples where SalEMA is doing better than SalCLSTM:  " + str(rate))
    #plt.show()

def plot_scatter (x1, y1, name1, x2, y2, name2, Xaxis_name, Yaxis_name):

    fig, ax = plt.subplots(figsize=(9,7))
    ax.scatter(x1, y1,  label=name1)
    ax.scatter(x2, y2, label=name2)
    legend = ax.legend(loc='lower right', shadow=False, fontsize='xx-large')
    plt.grid(True)

    plt.xlabel(Xaxis_name)
    plt.ylabel(Yaxis_name)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)



def load_files(path1, path2):
    data1 = np.asarray( np.load(path1) )
    data2 = np.asarray( np.load(path2) )

    return [data1, data2]

if __name__ == '__main__':
    main()
