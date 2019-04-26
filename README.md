# SalEMA

This work is an improvement on [VideoSalGAN](https://github.com/imatge-upc/saliency-2018-videosalgan).
In both of these works, the goal is to explore how a model trained on static images for the task of saliency prediction can be extended to do the same thing on videos. The video saliency dataset used for our experiments was the DHF1K.

The original architecture ([SalGAN](https://imatge-upc.github.io/saliency-salgan-2017/)) is trained on SALICON and optimized by a combination of two cost functions, binary cross entropy and adversarial loss. 
![image](https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/figs/fullarchitecture.jpg?token=AFOjyaH8cuBFWpldWWzo_TKVB-zekfxrks5Yc4NQwA%3D%3D)

The saliency generator consists of an encoder and a decoder part. Porting to PyTorch showed an unexplained loss in performance. For this reason, before fine-tuning on a dynamic dataset, we needed to set a new baseline of our ported pytorch model. In order to do this, we tried a few experiments and the best performance was achieved with 27 epochs over SALICON with data augmentation techniques applied. 

After setting a [baseline](https://github.com/juanjo3ns/SalBCE) we applied our temporal augmentation, which essentially was choosing a layer to act as the temporal state or add a new one. We experimented with two types of modifications: an exponential moving average of the temporal state and the addition of a ConvLSTM layer to act as the temporal state. We trained our architecture on [DHF1K](https://github.com/wenguanwang/DHF1K). Below is an schematic representing one example of our models, where the temporal state was added at the bottleneck.
![TemporalEDmodel](https://raw.githubusercontent.com/Linardos/VideoSalGAN-II/blob/gh-pages/TemporalEDmodel.jpg)

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

Our results indicate that the simple addition of EMA even without extra training does almost as well as a sophisticated ConvLSTM augmentation and even surpasses it after being fine-tuned. EMA essentially performs a smoothing over the frames of the video by averaging over the frames. A possible explanation for why this boosts performance in video saliency is that saliency tends to be relatively consistent across frames, with the exception of rapid movements. Another relevant point is that an averaging of the frames should have the effect of bringing the probabilities closer to the center, taking advantage of the center bias in that sense. It is likely that the ConvLSTM layer approximates a similar function.

We performed further experiments by tampering with the position of the EMA, but results were similar despite the placement. We also tried placing 2 EMAs simultaneously and setting their respective alpha values to 0.3, but this performed much worse.

| DHF1K	| AUC-J	| s-AUC	| NSS	| CC | SIM |
| ----- | ----- | ----- | --- | -- | --- |
|SalEMA30 (Bottleneck) |	0.883	| 0.734 |	2.144	| 0.400 |	0.276 |
|SalEMA61 (Output)	| 0.884	|0.737	|2.133	|0.399	|0.270|
|SalEMA54 (Decoder) |	0.883	|0.734|	2.149	|0.401|	0.276|
|SalEMA61 Tuned |	0.888	| 0.681	| 2.394|	0.438|	0.354|
|SalEMA7&54 (Encoder&Decoder)	| 0.828	| 0.561	| 1.403	| 0.366	| 0.344 |

Qualitative Results on video #664 (EMA shown to improve NSS a lot in this particular sample)

![QResults](https://github.com/Linardos/VideoSalGAN-II/blob/master/QResultsEMA.png)

Qualitative Results on video #601 (EMA shown to do worse on NSS in this particular sample)

![QResults](https://github.com/Linardos/VideoSalGAN-II/blob/master/QResultsCLSTM.png)
