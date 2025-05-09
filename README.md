# FedMAE


[![Static Badge](https://img.shields.io/badge/AAAI-XXXX-red?style=plastic&logo=AAAI&labelColor=%2386C166&color=grey)](https://iclr.cc/virtual/2024/poster/17446) | 
[![Static Badge](https://img.shields.io/badge/OpenReview-FedDAE-red?style=plastic&logo=OpenReivew&labelColor=%23FCFAF2&color=grey)](https://openreview.net/forum?id=Z91eH3ajOr) | 
[![Static Badge](https://img.shields.io/badge/arxiv-2408.08931-red?style=plastic&logo=arxiv&logoColor=white&labelColor=%23C73E3A&color=grey)](https://arxiv.org/abs/2408.08931)

> This project is the code and the supplementary of "**Personalized Federated Collaborative Filtering: A Variational AutoEncoder Approach**"

## Requirements

1. The code is implemented with `Python ~= 3.8` and `torch~=2.3.1+cu117`;
2. Other requirements can be installed by `pip install -r requirements.txt`.

## Quick Start

**Notice**: `FedDAE` was previously referred to as `FedMAE`, so terms like "MAE" may appear in the code.

1. Put datasets into the path `[parent_folder]/datasets/`;

2. For quick start, please run:
    ``````
    python main.py --alias FedMAE --dataset movielens --data_file ml-100k.dat \
        --lr 1e-3 --l2_reg 1e-5 --seed 0
    ``````

3. if you want to use the notice function `mail_notice`, please set your own keys.

## Thanks

In the implementation of this project, we referred to the code of [RecBole](https://github.com/RUCAIBox/RecBole) and [Tenrec](https://github.com/yuangh-x/2022-NIPS-Tenrec?tab=readme-ov-file), and we are grateful for their open-source contributions!



## Contact

- This project is free for academic usage. You can run it at your own risk.
- For any other purposes, please contact Mr. Zhiwei Li ([lizhw.cs@outlook.com](mailto:lizhw.cs@outlook.com))