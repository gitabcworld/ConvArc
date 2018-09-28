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

## Execute

| Channels | Dataset        | Within/Across | ARC/CONVARC   | LSTM  | Naive/Full    | Ref. Command | 
| -------- | -------------- | ------------- | ------------- | ----- | ------------- | ------------ |
| 1        | omniglot       | within        | arc           | lstm  | naive         | 1            |
| 1        | omniglot       | across        | arc           | lstm  | naive         | 2            |
| 1        | omniglot       | within        | arc           | lstm  | fullcontext   | 3            |
| 1        | omniglot       | across        | arc           | lstm  | fullcontext   | 4            |
| 3        | omniglot       | within        | arc           | lstm  | naive         | 5            |
| 3        | omniglot       | across        | arc           | lstm  | naive         | 6            |
| 3        | omniglot       | within        | arc           | lstm  | fullcontext   | 7            |
| 3        | omniglot       | across        | arc           | lstm  | fullcontext   | 8            |
| 1        | omniglot       | within        | convarc       | lstm  | naive         | 9            |
| 1        | omniglot       | across        | convarc       | lstm  | naive         | 10           |
| 1        | omniglot       | within        | convarc       | lstm  | fullcontext   | 11           |
| 1        | omniglot       | across        | convarc       | lstm  | fullcontext   | 12           |
| 3        | omniglot       | within        | convarc       | lstm  | naive         | 13           |
| 3        | omniglot       | across        | convarc       | lstm  | naive         | 14           |
| 3        | omniglot       | within        | convarc       | lstm  | fullcontext   | 15           |
| 3        | omniglot       | across        | convarc       | lstm  | fullcontext   | 16           |

Ref command 1: ```python main.py --nchannels 1 --save results/os_w/lstm_channel_1_arc_naive/ --isWithinAlphabets True --apply_wrn False --wrn_save results/os_w/lstm_channel_1_arc_naive/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_w/lstm_channel_1_arc_naive/ARCmodel.pt7 --naive_full_type Naive --naive_full_save_path results/os_w/lstm_channel_1_arc_naive/context.pt7```

Ref command 2: ```python main.py --nchannels 1 --save results/os_a/lstm_channel_1_arc_naive/ --isWithinAlphabets False --apply_wrn False --wrn_save results/os_a/lstm_channel_1_arc_naive/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_a/lstm_channel_1_arc_naive/ARCmodel.pt7 --naive_full_type Naive --naive_full_save_path results/os_a/lstm_channel_1_arc_naive/context.pt7```

Ref command 3: ```python main.py --nchannels 1 --save results/os_w/lstm_channel_1_arc_fullc/ --isWithinAlphabets True --apply_wrn False --wrn_save results/os_w/lstm_channel_1_arc_fullc/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_w/lstm_channel_1_arc_fullc/ARCmodel.pt7 --naive_full_type FullContext --naive_full_save_path results/os_w/lstm_channel_1_arc_fullc/context.pt7```

Ref command 4: ```python main.py --nchannels 1 --save results/os_a/lstm_channel_1_arc_fullc/ --isWithinAlphabets False --apply_wrn False --wrn_save results/os_a/lstm_channel_1_arc_fullc/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_a/lstm_channel_1_arc_fullc/ARCmodel.pt7 --naive_full_type FullContext --naive_full_save_path results/os_a/lstm_channel_1_arc_fullc/context.pt7```

Ref command 5: ```python main.py --nchannels 3 --save results/os_w/lstm_channel_3_arc_naive/ --isWithinAlphabets True --apply_wrn False --wrn_save results/os_w/lstm_channel_3_arc_naive/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_w/lstm_channel_1_arc_naive/ARCmodel.pt7 --naive_full_type Naive --naive_full_save_path results/os_w/lstm_channel_3_arc_naive/context.pt7```

Ref command 6: ```python main.py --nchannels 3 --save results/os_a/lstm_channel_3_arc_naive/ --isWithinAlphabets False --apply_wrn False --wrn_save results/os_a/lstm_channel_3_arc_naive/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_a/lstm_channel_1_arc_naive/ARCmodel.pt7 --naive_full_type Naive --naive_full_save_path results/os_a/lstm_channel_3_arc_naive/context.pt7```

Ref command 7: ```python main.py --nchannels 3 --save results/os_w/lstm_channel_3_arc_fullc/ --isWithinAlphabets True --apply_wrn False --wrn_save results/os_w/lstm_channel_3_arc_fullc/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_w/lstm_channel_1_arc_fullc/ARCmodel.pt7 --naive_full_type FullContext --naive_full_save_path results/os_w/lstm_channel_3_arc_fullc/context.pt7```

Ref command 8: ```python main.py --nchannels 3 --save results/os_a/lstm_channel_3_arc_fullc/ --isWithinAlphabets False --apply_wrn False --wrn_save results/os_a/lstm_channel_3_arc_fullc/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_a/lstm_channel_1_arc_fullc/ARCmodel.pt7 --naive_full_type FullContext --naive_full_save_path results/os_a/lstm_channel_3_arc_fullc/context.pt7```

Ref command 9: ```python main.py --nchannels 1 --save results/os_w/lstm_channel_1_carc_naive/ --isWithinAlphabets True --apply_wrn True --wrn_save results/os_w/lstm_channel_1_carc_naive/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_w/lstm_channel_1_carc_naive/ARCmodel.pt7 --naive_full_type Naive --naive_full_save_path results/os_w/lstm_channel_1_carc_naive/context.pt7```

Ref command 10: ```python main.py --nchannels 1 --save results/os_a/lstm_channel_1_carc_naive/ --isWithinAlphabets False --apply_wrn True --wrn_save results/os_a/lstm_channel_1_carc_naive/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_a/lstm_channel_1_carc_naive/ARCmodel.pt7 --naive_full_type Naive --naive_full_save_path results/os_a/lstm_channel_1_carc_naive/context.pt7```

Ref command 11: ```python main.py --nchannels 1 --save results/os_w/lstm_channel_1_carc_fullc/ --isWithinAlphabets True --apply_wrn True --wrn_save results/os_w/lstm_channel_1_carc_fullc/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_w/lstm_channel_1_carc_fullc/ARCmodel.pt7 --naive_full_type FullContext --naive_full_save_path results/os_w/lstm_channel_1_carc_fullc/context.pt7```

Ref command 12: ```python main.py --nchannels 1 --save results/os_a/lstm_channel_1_carc_fullc/ --isWithinAlphabets False --apply_wrn True --wrn_save results/os_a/lstm_channel_1_carc_fullc/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_a/lstm_channel_1_carc_fullc/ARCmodel.pt7 --naive_full_type FullContext --naive_full_save_path results/os_a/lstm_channel_1_carc_fullc/context.pt7```

Ref command 13: ```python main.py --nchannels 3 --save results/os_w/lstm_channel_3_carc_naive/ --isWithinAlphabets True --apply_wrn True --wrn_save results/os_w/lstm_channel_3_carc_naive/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_w/lstm_channel_1_carc_naive/ARCmodel.pt7 --naive_full_type Naive --naive_full_save_path results/os_w/lstm_channel_3_carc_naive/context.pt7```

Ref command 14: ```python main.py --nchannels 3 --save results/os_a/lstm_channel_3_carc_naive/ --isWithinAlphabets False --apply_wrn True --wrn_save results/os_a/lstm_channel_3_carc_naive/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_a/lstm_channel_1_carc_naive/ARCmodel.pt7 --naive_full_type Naive --naive_full_save_path results/os_a/lstm_channel_3_carc_naive/context.pt7```

Ref command 15: ```python main.py --nchannels 3 --save results/os_w/lstm_channel_3_carc_fullc/ --isWithinAlphabets True --apply_wrn True --wrn_save results/os_w/lstm_channel_3_carc_fullc/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_w/lstm_channel_1_carc_fullc/ARCmodel.pt7 --naive_full_type FullContext --naive_full_save_path results/os_w/lstm_channel_3_carc_fullc/context.pt7```

Ref command 16: ```python main.py --nchannels 3 --save results/os_a/lstm_channel_3_carc_fullc/ --isWithinAlphabets False --apply_wrn True --wrn_save results/os_a/lstm_channel_3_carc_fullc/ --arc_nchannels 64 --arc_attn_type LSTM --arc_save results/os_a/lstm_channel_1_carc_fullc/ARCmodel.pt7 --naive_full_type FullContext --naive_full_save_path results/os_a/lstm_channel_3_carc_fullc/context.pt7```


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