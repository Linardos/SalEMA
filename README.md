# VideoSalGAN-II

This work is an improvement on [VideoSalGAN](https://github.com/imatge-upc/saliency-2018-videosalgan).
In both of these works, the goal is to explore how a model trained on static images for the task of saliency prediction can be extended to do the same thing on videos. 

The original architecture ([SalGAN](https://imatge-upc.github.io/saliency-salgan-2017/)) is trained on SALICON and optimized by a combination of two cost functions, binary cross entropy and adversarial loss. 
![image](https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/figs/fullarchitecture.jpg?token=AFOjyaH8cuBFWpldWWzo_TKVB-zekfxrks5Yc4NQwA%3D%3D)

The saliency generator consists of an encoder and a decoder part. What we did is use these two parts with their respective pretrained weights and add a ConvLSTM cell in the middle, to extract temporally significant features. We trained our architecture on [DHF1K](https://github.com/wenguanwang/DHF1K). 
![salganmid](https://github.com/Linardos/VideoSalGAN-II/blob/master/SalGANmid.jpg)

For evaluation purposes we also tuned an instance of SalGAN on the DHF1K dataset. We evaluated both versions of SalGAN and our model, and found that our augmentation boosts performance on this dataset.

| DHF1K	| AUC-J	| s-AUC	| NSS	| CC | SIM |
| ----- | ----- | ----- | --- | -- | --- |
| SalGAN (Port) |	0.807 |	0.661 |	1.545	| 0.282 |	0.197 |
| SalGAN (Tuned)	| 0.874	| 0.645	| 2.226	| 0.395	| 0.338 |
| VideoSalGAN-II	| 0.915	| 0.745	| 3.173	| 0.556	| 0.440 |

