# ConvArc

This repo provides a Pytorch implementation fo the [Attentive Recurrent Comparators](https://arxiv.org/pdf/1703.00767.pdf) paper.

## Acknowledgements
Special thanks to Pranav Shyam and Sanyam Agarwal for their Theano and Pytorch implementation in which this work is based on. 
For more details:
https://github.com/pranv/ARC
https://github.com/sanyam5/arc-pytorch

## Cite
```
@article{DBLP:journals/corr/ShyamGD17,
  author    = {Pranav Shyam and
               Shubham Gupta and
               Ambedkar Dukkipati},
  title     = {Attentive Recurrent Comparators},
  journal   = {CoRR},
  volume    = {abs/1703.00767},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.00767},
  timestamp = {Wed, 07 Jun 2017 14:42:50 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/ShyamGD17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

## Install
```
conda create --name py3pytorch4 python=3.6
source activate py3pytorch4
conda install -c conda-forge opencv
pip install tensorboardX
conda install pytorch torchvision cuda80 -c pytorch
conda install scikit-learn
pip install torchnet
conda install -c conda-forge tqdm
conda install -c anaconda psutil 
conda install -c anaconda scikit-image
pip install nested_dict
pip install hickle
conda install -c anaconda graphviz
```


## Authors

* Albert Berenguel (@aberenguel) [Webpage](https://scholar.google.es/citations?user=HJx2fRsAAAAJ&hl=en)