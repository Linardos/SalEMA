# VideoSalGAN-II

This work is an improvement on [VideoSalGAN](https://github.com/imatge-upc/saliency-2018-videosalgan).
In both of these works, the goal is to explore how a model trained on static images for the task of saliency prediction can be extended to do the same thing on videos. 

The original architecture ([SalGAN](https://imatge-upc.github.io/saliency-salgan-2017/)) is trained on SALICON. Here 

| DHF1K	| AUC-J	| s-AUC	| NSS	| CC |	SIM |
| ------------- | ----- | ----- | ----- | ----- |
| SalGAN (Port) |	0.807 |	0.661 |	1.545	| 0.282 |	0.197 |
| SalGAN (Tuned)	| 0.874	| 0.645	| 2.226	| 0.395	| 0.338 |
| VideoSalGAN-II	| 0.915	| 0.745	| 3.173	| 0.556	| 0.440 |
