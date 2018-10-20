import torch
import numpy as np
import salgan_generator


# parameters for demo inference=================================================
PATH_LASAGNE_WEIGHTS = 'model_weights/gen_modelWeights0090.npz'
PATH_PYTORCH_WEIGHTS = 'model_weights/gen_model.pt'


def port_lasagne_weights(model, path_weights):
    """
    Port lasagne weights into numpy model

    args:
        model: salgan model in pytorch
        path_weights: path to original lasagne weights
    """
    # load weight file (dictionary)
    weights = np.load(path_weights)

    # get key values (name of parameters)
    keys = list(weights.iterkeys())

    # sort parameters based on name
    sort_weights = {}
    for m in range(54):
        for i, x in enumerate(list(weights.items())):
            if m==int(x[0].split('_')[1]):
                sort_weights[m] = x[1]

    # loop over model parameters and assign lasagne weights
    for i, py_param in enumerate(model.parameters()):

        w = sort_weights[i]
        if len(w.shape) ==4:
            # we need to flip filter order to get correct results in pytorch
            lasagne = torch.FloatTensor(np.copy(w[::-1,::-1,:,:]).astype(np.float32))
        else:
            lasagne = torch.FloatTensor(np.copy(w[::-1]).astype(np.float32))

        py_param.data = lasagne


if __name__ == '__main__':
    # init model
    model = salgan_generator.create_model()

    print("porting weights from lasagne, these might take a few seconds...")
    port_lasagne_weights(model, PATH_LASAGNE_WEIGHTS)

    # save
    torch.save(model.state_dict(), PATH_PYTORCH_WEIGHTS)
    print("done! Pytorch weights {}".format(PATH_PYTORCH_WEIGHTS))
