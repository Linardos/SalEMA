# SalEMA

SalEMA is a video saliency prediction network. It utilizes a moving average of convolutional states to produce state of the art results. The architecture has been trained on DHF1K.

## Publication
Find the pre-print version of our work on [arXiv](https://arxiv.org/abs/1907.01869) or check our friendly summary on [medium](https://medium.com/@linardos.akis/teaching-machines-where-to-look-on-videos-c7a685e2b136)

## Model

![TemporalEDmodel](https://raw.githubusercontent.com/Linardos/SalEMA/gh-pages/TemporalEDmodel.jpg)

* Download our best configuration of the SalEMA model [here (364MB)](https://imatge.upc.edu/web/sites/default/files/projects/saliency/public/VideoSalGAN-II/SalEMA30.pt)

Sample video (click to be redirected to youtube):

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/JNe6A7dszPw/0.jpg)](https://www.youtube.com/watch?v=JNe6A7dszPw)

## Installation

- Clone the repo:

```shell
git clone https://github.com/Linardos/SalEMA
```

- Install requirements
```shell
pip install -r requirements.txt
```

## Inference

You may use our pretrained model for inference on either of the 3 datasets: DHF1K [[link]](https://drive.google.com/file/d/1vfRKJloNSIczYEOVjB4zMK8r0k4VJuWk/view), Hollywood-2 [[link]](https://drive.google.com/file/d/1vfRKJloNSIczYEOVjB4zMK8r0k4VJuWk/view), UCF-sports [[link]](https://drive.google.com/drive/folders/1sW0tf9RQMO4RR7SyKhU8Kmbm4jwkFGpQ) or your own dataset so long as it follows a specific folder structure:

To perform inference on DHF1K validation set:

```shell
python inference.py -dataset=DHF1K -pt_model=SalEMA30.pt -alpha=0.1 -start=600 -end=700 -dst=/path/to/output -src=/path/to/DHF1K
```

To perform inference on Hollywood-2 or UCF-sports test set (because of the way the dataset is structured, it's convenient to use the same path for dst and src):

```shell
python inference.py -dataset=Hollywood-2 -pt_model=SalEMA30.pt -alpha=0.1 -dst=/path/to/Hollywood-2/testing -src=/path/to/Hollywood-2/testing
```

```shell
python inference.py -dataset=UCF-sports -pt_model=SalEMA30.pt -alpha=0.1 -dst=/path/to/UCF-sports/testing -src=/path/to/UCF-sports/testing
```

To perform inference on your own dataset make sure to follow a simple folder structure (one superfolder given as root at the input, which includes folders of frames) and use the tag "other":

```shell
python inference.py -dataset=other -alpha=0.1 -pt_model=SalEMA30.pt -dst=/path/to/output -src=/path/to/superfolder/frames
```

If your dataset follows a more quirky structure you might need to manipulate the data_loader source code.

## Training

To train on DHF1K using CUDA:

<!-- ```shell
python python train.py -dataset=DHF1K -pt_model=False -start=1 -end=600 -src=/imatge/lpanagiotis/work/DHF1K
``` -->

```shell
python train.py -dataset=DHF1K -pt_model=False -new_model=SalEMA -ema_loc=30 -start=1 -end=4 -src=/path/to/DHF1K -use_gpu='gpu' -epochs=7
```

To train on Hollywood-2, UCF-sports using CUDA. For fine-tuning a pretrained model, use a higher number of epochs, the training commences from the epoch number where it stopped on:


```shell
python train.py -dataset=Hollywood-2 -pt_model=SalEMA30.pt -new_model=SalEMA -ema_loc=30 -src=/path/to/Hollywood-2 -use_gpu='gpu' -epochs=10 -lr=0.0000001
```

