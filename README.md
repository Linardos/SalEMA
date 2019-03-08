# SalEMA

This work is an improvement on [VideoSalGAN](https://github.com/imatge-upc/saliency-2018-videosalgan).
In both of these works, the goal is to explore how a model trained on static images for the task of saliency prediction can be extended to do the same thing on videos. The video saliency dataset used for our experiments was the DHF1K.

The original architecture ([SalGAN](https://imatge-upc.github.io/saliency-salgan-2017/)) is trained on SALICON and optimized by a combination of two cost functions, binary cross entropy and adversarial loss. 
![image](https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/figs/fullarchitecture.jpg?token=AFOjyaH8cuBFWpldWWzo_TKVB-zekfxrks5Yc4NQwA%3D%3D)

The saliency generator consists of an encoder and a decoder part. Porting to PyTorch showed an unexplained loss in performance. For this reason, before fine-tuning on a dynamic dataset, we needed to set a new baseline of our ported pytorch model. In order to do this, we tried a few experiments and the best performance was achieved with 27 epochs over SALICON with data augmentation techniques applied. 

After setting a ([baseline](https://github.com/juanjo3ns/SalBCE)) we applied our temporal augmentation, which essentially was choosing a layer to act as the temporal state or add a new one. We experimented with two types of modifications: an exponential moving average of the temporal state and the addition of a ConvLSTM layer to act as the temporal state. We trained our architecture on [DHF1K](https://github.com/wenguanwang/DHF1K). Below is an schematic representing one example of our models, where the temporal state was added at the bottleneck.
![TemporalEDmodel](https://github.com/Linardos/VideoSalGAN-II/blob/master/TemporalEDmodel.png)

Evaluation on DHF1K showed that our augmentations improve performance over the baseline. 

| DHF1K	| AUC-J	| s-AUC	| NSS	| CC | SIM |
| ----- | ----- | ----- | --- | -- | --- |
| SalGAN Port | 0.801	| 0.652	| 1.437	| 0.267	| 0.192 |
| ([SalBCE (baseline)](https://github.com/juanjo3ns/SalBCE))| 0.874	| 0.724	| 2.047	| 0.382	| 0.268 |
| SalCLSTM30 || | 	| 	|  |
| SalEMA30 | 0.883 |	0.685 |	2.402 |	0.435 |	0.349 |
| SalEMA30 (with dropout) | 0.886	| 0.690	| 2.495	| 0.450	| 0.360 |
