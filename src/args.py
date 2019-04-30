import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='Video saliency prediction')

    parser.add_argument('-use_gpu', dest='use_gpu', default=True)
    parser.add_argument('-dst', dest='dst', default="/imatge/lpanagiotis/work/DHF1K", help="Add root path to output predictions to.")
    parser.add_argument('-src', dest='src', default="/imatge/lpanagiotis/work/DHF1K/frames", help="Add root path to  predictions to.")
    # Dataset
    parser.add_argument('-dataset', dest='dataset', default='DHF1K')
    parser.add_argument('-start', dest='start', default=1)
    parser.add_argument('-end', dest='end', default=700)

    # Model
    parser.add_argument('-pretrained_model', dest='pretrained_model', default='SalEMA30.pt', help="Input path to stored model.")
    parser.add_argument('-new_model', dest='new_model', default='SalEMA', help="Input name to use on newly trained model (Don't include extension or number for the recurrence placement; they will be added automatically")

    # Args for EMA
    parser.add_argument('-alpha', dest='alpha', default=0.1, help='Input value for alpha.')
    parser.add_argument('-ema_loc', dest='ema_loc', default=True, help='Input number of layer to place EMA on')
    parser.add_argument('-dropout', dest='dropout', default=True, help='Boolean value to add dropout on SalEMA')
    parser.add_argument('-residual', dest='residual', default=False, help='Boolean value to add a residual connection on SalEMA')
    parser.add_argument('-double_ema', dest='double_ema', default=False, help='Boolean value to activate two EMAs simultaneously')


    return parser

if __name__ =="__main__":

    parser = get_parser()
    args_dict = parser.parse_args()
    print(args_dict)
