import argparse

def get_inference_parser():

    parser = argparse.ArgumentParser(description="Video saliency prediction. Use this script to get saliency maps for your videos.  If you're using your own dataset make sure to follow the same folder structure as DHF1K: Videos divided into numbered folders [0001, 0002 etc] that include their extracted frames [also numbered 0001.png, 0002.png etc].")

    parser.add_argument('-use_gpu', dest='use_gpu', default=True, help="Boolean value, set to True if you are using CUDA.")
    parser.add_argument('-dst', dest='dst', default="/imatge/lpanagiotis/work/DHF1K", help="Add root path to output predictions to.")
    parser.add_argument('-src', dest='src', default="/imatge/lpanagiotis/work/DHF1K/frames", help="Add root path to your dataset.")

    # Dataset
    parser.add_argument('-dataset', dest='dataset', default='DHF1K', help="Name of the dataset to be inferred. These can be 'DHF1K', 'Hollywood-2', 'UCF-sports'. If you wish to use your own dataset input 'other'.")
    parser.add_argument('-start', dest='start', default=1, help='Define the video from which you would wish to start inferrence (number of folder).')
    parser.add_argument('-end', dest='end', default=700, help='Define the video at which you would wish to end inferrence (number of folder).')

    # Model
    parser.add_argument('-pt_model', dest='pt_model', default='SalEMA30.pt', help="Input path to pretrained model.")
    #parser.add_argument('-new_model', dest='new_model', default='SalEMA', help="Input name to use on newly trained model (Don't include extension or number for the recurrence placement; they will be added automatically")

    # Args for EMA
    parser.add_argument('-alpha', dest='alpha', default=0.1, help='Input value for alpha.')
    parser.add_argument('-ema_loc', dest='ema_loc', default=True, help='Input number of layer to place EMA on')
    parser.add_argument('-dropout', dest='dropout', default=True, help='Boolean value set to True if model includes dropout.')
    parser.add_argument('-residual', dest='residual', default=False, help='Boolean value set to True if model includes residual connection on SalEMA')
    parser.add_argument('-double_ema', dest='double_ema', default=False, help='Boolean value set to True to activate two EMAs simultaneously')


    return parser

if __name__ =="__main__":

    parser = get_inference_parser()
    args_dict = parser.parse_args()
    print(args_dict)
