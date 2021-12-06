# 字节跳动安全AI挑战赛-把杰泥牛逼打在公屏上-Writeup & Reproduce

## **1. 环境依赖**

- Python 3.7.0
- numpy 1.19.5
- pandas 0.24.2
- scikit-learn 0.24.2
- tqdm 4.46.1
- gensim 3.8.3
- lightgbm 3.3.1
  
## **2. 目录结构**

```
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements
├── train.sh, script for preparing training data and training models, including pretrained models
├── code
│   ├── train.py main code
├── data
│   ├── data file
├── submission
│   ├── submission file
```

## **3. 运行流程**

- 将数据集放至data目录下
- 安装环境：sh init.sh
- 模型训练、预测：sh train.sh

## **4. 模型及特征**
- 模型：[lightgbm](http://papers.nips.cc/paper/6907-a-highly-efficient-gradient-boosting-decision-tree.pdf)
- 特征：
    - 用户侧特征：
       - 账户本身的基础特征
       - 账户本身的特征计数统计
       - 粉丝量、关注量、发帖量、被点赞量 除法交叉
       - 登录时间、注册时间 减法交叉

    - 请求侧特征：
      - 点赞、关注基础特征
      - 机型、ip、app_version、app_channel统计
      - 聚合特征1，在每个请求ip下有多少不同的用户，用户请求的所有ip的用户数的均值和方差
      - 聚合特征2， 用户请求时间的均值方差等等
      - w2v特征， 每个用户的请求ip序列、机型序列等等建模

## **5. 方案说明**

- 本次方案采用单模型LGB模型进行训练预测，主要从两张表中提取特征，用户基础信息表可以对账户本身的基础特征进行刻画，用户请求表是用户请求行为的记录，可针对此表刻画用户的行为形象，采用的方案是以每一条请求记录作为一条数据进行建模，最终对于每个用户取请求行为的预测概率最大值作为预测结果；

- 特征方案也从上述两个方面展开，基于账户本身基础特征，可以做这些类别特征的计数统计（'user_freq_ip', 'user_profile', 'user_name', 'user_register_time', 'user_least_login_time',  'user_register_type', 'user_register_app', 'user_least_login_app'这些数值列在数据中出现了多少次，用value_counts()计算）、对于粉丝量等数值特征可以做除法的交叉（user_post_like_num/user_post_num， user_post_like_num/user_follow_num， user_post_like_num/user_fans_num，user_post_num/user_follow_num，user_post_num/user_fans_num，user_follow_num， user_fans_num）、登录时间和注册时间特征可以做减法交叉（user_least_login_time-user_register_time），基于请求行为，我们可以对机型、ip、app_version、app_channel做频数统计（'request_model_id', 'request_ip','request_device_type', 'request_app_version'， 'request_app_channel' 这些数值列在数据中出现了多少次,用value_counts()计算）对于用户的请求行为序列，我们可以构建w2v特征(把用户请求行为序列看成句子，行为看作词，训练Word2Vec模型得到每个行为的表征)，对于用户的请求时间，我们可以计算请求时间的均值方差、请求时间间隔的统计特征等等；
          
- 模型训练采用传统分层五折交叉验证，本方案未对模型进行参数调参以及输出后处理。

## **6. 一些尝试**

- 本方案的解题思路与题目所要求的可能并不匹配，尝试使用一些半监督学习的方法，包括伪标签、标签传播等来更加充分利用无标签样本，但是几乎都没有收益；
- 如何更好地利用无标签样本可能是本题的解题关键，期待其他选手的分享。

## **7. 算法性能**

- 资源配置：cpu i7 16G内存
- 总计耗时：约为60分钟

## **8. 相关文献**
* Ke G, Meng Q, Finley T, et al. Lightgbm: A highly efficient gradient boosting decision tree[J]. Advances in neural information processing systems, 2017, 30: 3146-3154.
* Mikolov T, Chen K, Corrado G, et al. Efficient estimation of word representations in vector space[J]. arXiv preprint arXiv:1301.3781, 2013.

