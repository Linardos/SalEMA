# SalEMA

Download our best configuration of the SalEMA model [here](https://imatge.upc.edu/web/sites/default/files/projects/saliency/public/VideoSalGAN-II/SalEMA30.pt)


This work is an improvement on [VideoSalGAN](https://github.com/imatge-upc/saliency-2018-videosalgan).
In both of these works, the goal is to explore how a model trained on static images for the task of saliency prediction can be extended to do the same thing on videos. The video saliency dataset used for our experiments was the DHF1K.

![TemporalEDmodel](https://raw.githubusercontent.com/Linardos/SalEMA/gh-pages/TemporalEDmodel.jpg)

![QResults](https://raw.githubusercontent.com/Linardos/SalEMA/gh-pages/QResultsEMA.png)

## Inference

An example of how to perform inference on DHF1K validation set:
```python inference.py -dataset=DHF1K -pretrained_model=SalEMA30A.pt -start=600 -end=700 -dst=/path/to/output -src=/path/to/DHF1K/frames```
