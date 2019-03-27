# MAAN
Pytorch implementation of paper "Marginalized Average Attentional Network for Weakly-Supervised Learning" (ICLR 2019).

[[PDF]](https://openreview.net/pdf?id=HkljioCcFQ)

Code is tested with **Pytorch 1.0** + **Python 3.6**. 

A simple test for the marginalized average aggregation (MAA) layer is provided with: 
```
python MAA.py
```

The aggregator is initially designed to aggregate video features, i.e. input is a 3D tensor : *Batch Size* X *T* X *N*

But it can easily be adapted to process image feature map, you only need to reshape 4D tensors into 3D tensors.

If our project is helpful for your research, please consider citing : 
``` 
@inproceedings{yuan2018marginalized,
          title={Marginalized Average Attentional Network for Weakly-Supervised Learning},
          author={Yuan, Yuan and Lyu, Yueming and Shen, Xi and Tsang, Ivor W and Yeung, Dit-Yan},
          booktitle={International Conference on Learning Representations (ICLR)},
          year={2019}
        }
```

