# SalEMA

This work is an improvement on [VideoSalGAN](https://github.com/imatge-upc/saliency-2018-videosalgan).
In both of these works, the goal is to explore how a model trained on static images for the task of saliency prediction can be extended to do the same thing on videos. The video saliency dataset used for our experiments was the DHF1K.

The original architecture ([SalGAN](https://imatge-upc.github.io/saliency-salgan-2017/)) is trained on SALICON and optimized by a combination of two cost functions, binary cross entropy and adversarial loss. 
![image](https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/figs/fullarchitecture.jpg?token=AFOjyaH8cuBFWpldWWzo_TKVB-zekfxrks5Yc4NQwA%3D%3D)

The saliency generator consists of an encoder and a decoder part. Porting to PyTorch showed an unexplained loss in performance. For this reason, before fine-tuning on a dynamic dataset, we needed to set a new baseline of our ported pytorch model. In order to do this, we tried a few experiments and the best performance was achieved with 27 epochs over SALICON with data augmentation techniques applied. 

After setting a [baseline](https://github.com/juanjo3ns/SalBCE) we applied our temporal augmentation, which essentially was choosing a layer to act as the temporal state or add a new one. We experimented with two types of modifications: an exponential moving average of the temporal state and the addition of a ConvLSTM layer to act as the temporal state. We trained our architecture on [DHF1K](https://github.com/wenguanwang/DHF1K). Below is an schematic representing one example of our models, where the temporal state was added at the bottleneck.
![TemporalEDmodel](https://raw.githubusercontent.com/Linardos/SalEMA/gh-pages/TemporalEDmodel.jpg)

Evaluation on DHF1K showed that our augmentations improve performance over the baseline. Note that in all EMA models, the alpha value is fixed at 0.1.

| DHF1K	| AUC-J	| s-AUC	| NSS	| CC | SIM |
| ----- | ----- | ----- | --- | -- | --- |
| SalGAN Port | 0.801	| 0.652	| 1.437	| 0.267	| 0.192 |
| [SalBCE (baseline)](https://github.com/juanjo3ns/SalBCE)| 0.874	| 0.724	| 2.047	| 0.382	| 0.268 |
| SalCLSTM30 | 0.887 | 0.693 | 2.364 |0.435|	0.322|
| SalEMA30 (no extra training) |	0.883	| 0.734 |	2.144	| 0.400 |	0.276 |
| SalEMA30 (fine-tuned after the addition of EMA) | 0.883 |	0.685 |	2.402 |	0.435 |	0.349 |
| SalEMA30 (with dropout) | 0.886	| 0.690	| 2.495	| 0.450	| 0.360 |
| SalEMA30R (with skip connection) |	0.875 |	0.670 |	2.274	| 0.415	| 0.339 |

Qualitative Results on video #664 (EMA shown to improve NSS a lot in this particular sample)

![QResults](https://raw.githubusercontent.com/Linardos/SalEMA/gh-pages/QResultsEMA.png)

Qualitative Results on video #601 (EMA shown to do worse on NSS in this particular sample)

![QResults](https://raw.githubusercontent.com/Linardos/SalEMA/gh-pages/QResultsCLSTM.png)


## Model

Download our best configuration of the SalEMA model [here](https://imatge.upc.edu/web/sites/default/files/projects/saliency/public/VideoSalGAN-II/SalEMA30.pt)

## Installation

- Clone the repo:

```shell
git clone https://github.com/Linardos/SalEMA
```

- Install requirements ```pip install -r requirements.txt``` 
- Install [PyTorch 1.0](http://pytorch.org/):

```shell
pip3 install torch torchvision
```

## Inference

You may use our pretrained model for inference on either of the 3 datasets: DHF1K [[link]](https://drive.google.com/file/d/1vfRKJloNSIczYEOVjB4zMK8r0k4VJuWk/view), Hollywood-2 [[link]](https://drive.google.com/file/d/1vfRKJloNSIczYEOVjB4zMK8r0k4VJuWk/view), UCF-sports [[link]](https://drive.google.com/drive/folders/1sW0tf9RQMO4RR7SyKhU8Kmbm4jwkFGpQ):

To perform inference on DHF1K validation set:

```shell
python inference.py -dataset=DHF1K -pt_model=SalEMA30.pt -start=600 -end=700 -dst=/path/to/output -src=/path/to/DHF1K/frames
```

To perform inference on Hollywood-2 or UCF-sports test set (because of the way the dataset is structured, it's convenient to use the same path for dst and src):

```shell
python inference.py -dataset=Hollywood-2 -pt_model=SalEMA30.pt -dst=/path/to/Hollywood-2/testing -src=/path/to/Hollywood-2/testing
```

```shell
python inference.py -dataset=UCF-sports -pt_model=SalEMA30.pt -dst=/path/to/UCF-sports/testing -src=/path/to/UCF-sports/testing
```

To perform inference on your own dataset make sure to follow the same structure as DHF1K (numbered folders followed by numbered frames):

```shell
python inference.py -dataset=other -pt_model=SalEMA30.pt -dst=/path/to/output -src=/path/to/your_dataset/frames
```

