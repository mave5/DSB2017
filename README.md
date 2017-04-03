# DSB2017

## Kaggle Data Science Bowl 2017 Lung Cancer Detection



---------------------

### Scattered DSB Nodes

These [plots](https://github.com/mravendi/DSB2017/tree/master/figs) are the scattered nodule candidates accross the lung volume for the DSB 2017 dataset. 
To obtain the candidates, first a 3D CNN was trained on 64 * 64 * 64 chunks from LUNA positive and negative candidates with 
quantized diameter as the output. After training, we deploy the 3D CNN on chunks of DSB 2017 dataset. 
Both dataset are re-sampled to 1mm * 1 mm * 1 mm. 


In the figures, the legends correspond to the quantized nodule diameter (step=4 mm). 

For instance, 
- nodule size < 4 mm is considered zero/non-nodule, not plotted
- 4 mm < nodule size < 8 mm is shown with legned 1.
- 8 mm < nodule size < 12 mm is shown with legned 2.
- 12 mm < nodule size < 16 mm is shown with legned 3.
- 16 mm < nodule size < 20 mm is shown with legned 4.

![fig.1](https://github.com/mravendi/DSB2017/blob/master/figs/90d6324d7006a3d142ee1884279dcf9b.jpg)
![fig.2](https://github.com/mravendi/DSB2017/blob/master/figs/90e5f4780b2f05136ff5f776a5cbc2af.jpg)
![fig.3](https://github.com/mravendi/DSB2017/blob/master/figs/8ed68f2dbf103a4bc0fd8708d8c1ac93.jpg)
![fig.4](https://github.com/mravendi/DSB2017/blob/master/figs/7faa456389e1ffde464819d0b1360188.jpg)
