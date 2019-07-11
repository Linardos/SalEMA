# SalEMA

SalEMA is a video saliency prediction network. It utilizes a moving average of convolutional states to produce state of the art results according to [this benchmark](https://mmcheng.net/videosal/) on DHF1K, Hollywwod-2 and UCF Sports (July 2019). The model has been trained on the [DHF1K dataset](https://github.com/wenguanwang/DHF1K). 

# Abstract

This paper investigates modifying an existing neural network architecture for static saliency prediction using two types of recurrences that integrate information from the temporal domain. The first modification is the addition of a ConvLSTM within the architecture, while the second is a conceptually simple exponential moving average of an internal convolutional state. We use weights pre-trained on the SALICON dataset and fine-tune our model on DHF1K. Our results show that both modifications achieve state-of-the-art results and produce similar saliency maps. 

## Publication

Find the extended pre-print version of our work on [arXiv](https://arxiv.org/abs/1907.01869). 

Please cite with the following Bibtex code:

```
@InProceedings{Linardos2019,
author = {Linardos, Panagiotis and Mohedano, Eva and Nieto, Juan Jose and McGuinness, Kevin and Giro-i-Nieto, Xavier and O'Connor, Noel E.},
title = {Temporal Recurrences for Video Saliency Prediction},
booktitle = {British Machine Vision Conference (BMVC)},
month = {September},
year = {2019}
}
```

You may also want to refer to our publication with the more human-friendly Chicago style:

*Panagiotis Linardos, Eva Mohedano, Juan Jose Nieto, Kevin McGuinness, Xavier Giro-i-Nieto and Noel E. O'Connor. "Temporal Recurrences for Video Saliency Prediction." BMVC 2019.*

## Results

Qualitative results:
![QResults](https://raw.githubusercontent.com/Linardos/SalEMA/gh-pages/QResultsEMA.png)

Sample video (click to be redirected to youtube):
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/JNe6A7dszPw/0.jpg)](https://www.youtube.com/watch?v=JNe6A7dszPw)

## Model

![TemporalEDmodel](https://raw.githubusercontent.com/Linardos/SalEMA/gh-pages/TemporalEDmodel.jpg)

Download our best configuration of the SalEMA model [here](https://imatge.upc.edu/web/sites/default/files/projects/saliency/public/VideoSalGAN-II/SalEMA30.pt) (364MB)

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

