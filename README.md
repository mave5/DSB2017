# DSB2017

## Kaggle Data Science Bowl 2017 Lung Cancer Detection



---------------------

### Scattered DSB Nodes

These [figures](https://github.com/mravendi/DSB2017/tree/master/figs) are the scattered nodule candidates accross the long volume for the DSB 2017 dataset. 
To obtain the candidates, first a 3D CNN was trained on 64 * 64 * 64 chunks from LUNA positive and negative candidates with 
quantized diameter as the output. After training, we deploy the 3D CNN on chunks of DSB 2017 dataset. 
Both dataset are re-sampled to 1mm * 1 mm * 1 mm. 


In the figures, the legends correspond to the quantized nodule diameter (step=4 mm).
![fig.1](https://github.com/mravendi/DSB2017/blob/master/figs/90d6324d7006a3d142ee1884279dcf9b.jpg)
![fig.2](https://github.com/mravendi/DSB2017/blob/master/figs/90e5f4780b2f05136ff5f776a5cbc2af.jpg)
