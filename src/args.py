import argparse

def get_inference_parser():

    parser = argparse.ArgumentParser(description="Video saliency prediction. Use this script to get saliency maps for your videos.  If you're using your own dataset make sure to follow the same folder structure as DHF1K: Videos divided into numbered folders [0001, 0002 etc] that include their extracted frames [also numbered 0001.png, 0002.png etc].")

    parser.add_argument('-use_gpu', dest='use_gpu', default=True, type=bool, help="Boolean value, set to True if you are using CUDA.")
    parser.add_argument('-dst', dest='dst', default="/imatge/lpanagiotis/work/DHF1K", help="Add root path to output predictions to.")
    parser.add_argument('-src', dest='src', default="/imatge/lpanagiotis/work/DHF1K", help="Add root path to your dataset.")

    # Dataset
    parser.add_argument('-dataset', dest='dataset', default='DHF1K', help="Name of the dataset to be inferred. These can be 'DHF1K', 'Hollywood-2', 'UCF-sports'. If you wish to use your own dataset input 'other'.")
    parser.add_argument('-start', dest='start', default=1, type=int, help='Define the video from which you would wish to start inferrence (number of folder).')
    parser.add_argument('-end', dest='end', default=700,type=int, help='Define the video at which you would wish to end inferrence (number of folder).')

    # Model
    parser.add_argument('-pt_model', dest='pt_model', default='SalEMA30.pt', help="Input path to pretrained model.")
    #parser.add_argument('-new_model', dest='new_model', default='SalEMA', help="Input name to use on newly trained model (Don't include extension or number for the recurrence placement; they will be added automatically")

    # Args for EMA
    parser.add_argument('-alpha', dest='alpha', default=None, type=float, help='Input value for alpha.')
    parser.add_argument('-ema_loc', dest='ema_loc', default=30, type=int, help='Input number of layer to place EMA on')
    parser.add_argument('-dropout', dest='dropout', default=True, type=bool, help='Boolean value set to True if model includes dropout.')
    parser.add_argument('-residual', dest='residual', default=False, type=bool, help='Boolean value set to True if model includes residual connection on SalEMA')
    parser.add_argument('-double_ema', dest='double_ema', default=False, type=bool, help='Boolean value set to True to activate two EMAs simultaneously')


    return parser

def get_training_parser():

    parser = argparse.ArgumentParser(description="Video saliency prediction. Use this script to train the model. If you want to train on a dataset other than DHF1K, Hollywood-2 or UCF-sports make sure it follows the same folder structure as DHF1K: Videos divided into numbered folders [0001, 0002 etc] that include their extracted frames [also numbered 0001.png, 0002.png etc]. Otherwise you will need to construct your own data loader and manipulate the source code accordingly.")

    parser.add_argument('-use_gpu', dest='use_gpu', default='parallel', help="If you are using cuda set to 'gpu'. If you want to use the DataParallel pytorch module set to 'parallel'. Otherwise, set to 'cpu'.")
    parser.add_argument('-src', dest='src', default="/imatge/lpanagiotis/work/DHF1K", help="Add root path to your dataset.")

    # Dataset
    parser.add_argument('-dataset', dest='dataset', default='DHF1K', help="Name of the dataset to train model with. These can be 'DHF1K', 'Hollywood-2', 'UCF-sports'. If you wish to use your own dataset input 'other'.")
    parser.add_argument('-start', dest='start', default=1, type=int, help='Define the video from which you would wish to start inferrence (number of folder).')
    parser.add_argument('-end', dest='end', default=700,type=int, help='Define the video at which you would wish to end inferrence (number of folder).')

    # Training
    parser.add_argument('-lr', dest='lr', default=0.0000001, type=float, help='Learning rate used when optimizing the parameters. Note that the alpha parameter is using a learning rate fixed at 0.1 as its importance is higher than any other individual parameter.')
    parser.add_argument('-epochs', dest='epochs', default=7, type=int, help='Number of epochs to run. If loading a pretrained model make sure to set a number higher than the already trained number of epochs; the model will continue training from the epoch at which it stopped.')
    parser.add_argument('-val_perc', dest='val_perc', default=0, type=float, help='Percentage to use as validation set for the loss metric.')

    # Model
    parser.add_argument('-pt_model', dest='pt_model', default='SalEMA30.pt', help="Input path to a pretrained model.")
    parser.add_argument('-new_model', dest='new_model', default='SalEMA', help="Input name to use on newly trained model (Don't include extension or number for the recurrence placement; they will be added automatically")

    # Args for CLSTM
    parser.add_argument('-thaw', dest='thaw', default=False, help='Parameter to use gradual thawing on the ConvLSTM implementation.')

    # Args for EMA
    #parser.add_argument('-alpha', dest='alpha', default=None, help='Input value for alpha. Set to None in order for the model to learn alpha (recommended).')
    parser.add_argument('-ema_loc', dest='ema_loc', default=30, type=int, help='Input number of layer to place EMA on')
    parser.add_argument('-dropout', dest='dropout', default=True, type=bool, help='Boolean value set to True if model includes dropout.')
    parser.add_argument('-residual', dest='residual', default=False, type=bool, help='Boolean value set to True if model includes residual connection on SalEMA')
    parser.add_argument('-double_ema', dest='double_ema', default=False, help="Set to False if you're not using two EMAs. If you want to use2 EMAs, use the number upon which you want the second ema to be located at")


    return parser

if __name__ =="__main__":

    parser = get_inference_parser()
    args_dict = parser.parse_args()
    print(args_dict)
