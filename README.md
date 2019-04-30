# Pytorch implementation of SalGAN

This repository contains a Pytorch implementation of SalGAN, a [script](port_weights.py) to port the pre-trained weights from lasagne to pytorch, and the already ported [weights](model_weights/gen_model.pt) in Pytorch.

To import the model in Pytorch, run:
```
import torch
import salgan_generator

model = salgan_generator.create_model()
model.load_state_dict(torch.load('model_weights/gen_model.pt'))
```

The [main.py](main.py) script contains an example to run an inference of SalGAN to an image:

Original:
![](sample_images/gnomo.jpeg?raw=true)

Saliency:
![](sample_saliency/gnomo.jpeg?raw=true)


Enjoy :-)
