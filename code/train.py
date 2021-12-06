import pandas as pd
import json
from tqdm import tqdm
import warnings
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import re
import os
import sklearn
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec

warnings.filterwarnings('ignore')

# 数据读取

DATA_PATH = './data/'

train_base_1 = pd.read_csv(DATA_PATH + '3-data_feature2_date1.txt')
train_base_2 = pd.read_csv(DATA_PATH + '4-data_feature2_date2.txt')

base = pd.concat([train_base_1, train_base_2])
del train_base_1, train_base_2

# 基础信息表去重
base.drop_duplicates('user', inplace=True)

# count统计
cat_cols = ['user_freq_ip', 'user_profile', 'user_name', 'user_register_time', 'user_least_login_time',  'user_register_type', 'user_register_app', 'user_least_login_app']
for f in cat_cols:
    base[f + '_base_count'] = base[f].map(base[f].value_counts())

# 编码转换
for col in ['user_freq_ip', 'user_profile', 'user_name',]:
    lb = LabelEncoder()
    base[col] = lb.fit_transform(base[col])

train_requests_1 = pd.read_csv(DATA_PATH + '1-data_feature1_date1.txt')
train_requests_2 = pd.read_csv(DATA_PATH + '2-data_feature1_date2.txt')
requests = pd.concat([train_requests_1, train_requests_2])
del train_requests_1, train_requests_2

requests.rename(columns={'request_user': 'user'}, inplace=True)

test = pd.read_csv(DATA_PATH + 'to_prediction_model.csv', header=None)
test.columns = ['user']

requests_1 = requests[~requests['request_label'].isna()]
user_list2 = list(test.user.unique())
print(len(user_list2))
user_list1 = list(requests_1.user.unique())
print(len(user_list1))
user_list = user_list2 + user_list1
print(len(user_list))

del requests_1, test

# 拼接基础表，requests表按时间排序

data = requests
data.drop(['request_id'], axis=1, inplace=True)
data.sort_values('request_time', inplace=True)
data = pd.merge(data, base, on='user', how='left')
del base


# 获取聚合特征
def get_agg_features(data):
    for f in tqdm(
            ['request_ip', 'request_model_id', 'request_device_type', 'request_app_version', 'request_app_channel',
             'request_time', 'request_target'
             ]):
        df_temp = requests[['user', f]]
        df_temp.drop_duplicates(inplace=True)
        df_temp = df_temp.groupby([f])['user'].nunique().reset_index()
        df_temp.columns = [f, 'uin_count']

        df_temp2 = requests[['user', f]]
        df_temp2.drop_duplicates(inplace=True)
        df_temp = df_temp2.merge(df_temp, how='left')

        df_temp = df_temp.groupby(['user'])['uin_count'].agg({
            '{}_uin_count_mean'.format(f): 'mean',
            '{}_uin_count_std'.format(f): 'std',

        }).reset_index()
        data = data.merge(df_temp, how='left')
    return data


data = get_agg_features(data)

# 获取W2V特征


def emb(df, f1, f2):
    emb_size = 16
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=10, min_count=1, sg=0, hs=0, seed=1, iter=5)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv.vocab:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    return tmp


if not os.path.exists(DATA_PATH + 'ip_df_global.pkl'):
    action_df = emb(requests, f1='user', f2='request_ip')
    action_df.to_pickle(DATA_PATH + 'ip_df_global.pkl')
else:
    action_df = pd.read_pickle(DATA_PATH + 'ip_df_global.pkl')

data = pd.merge(data, action_df, on='user', how='left')

if not os.path.exists(DATA_PATH + 'model_df_global.pkl'):
    action_df = emb(requests, f1='user', f2='request_model_id')
    action_df.to_pickle(DATA_PATH + 'model_df_global.pkl')
else:
    action_df = pd.read_pickle(DATA_PATH + 'model_df_global.pkl')

data = pd.merge(data, action_df, on='user', how='left')

if not os.path.exists(DATA_PATH + 'request_target_global.pkl'):
    action_df = emb(requests, f1='user', f2='request_target')
    action_df.to_pickle(DATA_PATH + 'request_target_global.pkl')
else:
    action_df = pd.read_pickle(DATA_PATH + 'request_target_global.pkl')

data = pd.merge(data, action_df, on='user', how='left')

if not os.path.exists(DATA_PATH + 'request_device_type_global.pkl'):
    action_df = emb(requests, f1='user', f2='request_device_type')
    action_df.to_pickle(DATA_PATH + 'request_device_type_global.pkl')
else:
    action_df = pd.read_pickle(DATA_PATH + 'request_device_type_global.pkl')

data = pd.merge(data, action_df, on='user', how='left')

if not os.path.exists(DATA_PATH + 'request_app_channel_global.pkl'):
    action_df = emb(requests, f1='user', f2='request_app_channel')
    action_df.to_pickle(DATA_PATH + 'request_app_channel_global.pkl')
else:
    action_df = pd.read_pickle(DATA_PATH + 'request_app_channel_global.pkl')

data = pd.merge(data, action_df, on='user', how='left')

# 基础特征工程

del action_df
data = data[data['user'].isin(user_list)]

data['num_ratio1'] = data['user_post_like_num'] / data['user_post_num']
data['num_ratio2'] = data['user_post_like_num'] / data['user_follow_num']
data['num_ratio3'] = data['user_post_like_num'] / data['user_fans_num']

data['num_ratio4'] = data['user_post_num'] / data['user_follow_num']
data['num_ratio5'] = data['user_post_num'] / data['user_fans_num']
data['num_ratio6'] = data['user_follow_num'] / data['user_fans_num']

data['time_diff'] = data['user_least_login_time'] - data['user_register_time']

f = 'request_model_id'
t = data.groupby(['user'])[f].agg([
    ('request_{}_count'.format(f), 'count'),
    ('request_{}_nunique'.format(f), 'nunique'),
]).reset_index()
data = pd.merge(data, t, on='user', how='left')
data['request_request_model_id_count'] = data['request_request_model_id_count'].fillna(0)

for f in ['request_app_channel', 'request_app_version', 'request_device_type', 'request_target', 'request_ip']:
    t = data.groupby(['user'])[f].agg([
        ('request_{}_nunique'.format(f), 'nunique'),
    ]).reset_index()
    data = pd.merge(data, t, on='user', how='left')

f = 'request_time'
t = data.groupby(['user'])[f].agg([
    ('{}_mean'.format(f), 'mean'),
    ('{}_std'.format(f), 'std'),
    ('{}_max'.format(f), 'max'),
    ('{}_min'.format(f), 'min'),
    ('{}_range_max'.format(f), lambda x: x.diff().max()),
    ('{}_range_min'.format(f), lambda x: x.diff().min()),
]).reset_index()
data = pd.merge(data, t, on='user', how='left')

data['time_diff2'] = data['request_time_min'] - data['user_register_time']
data['time_diff3'] = data['request_time_max'] - data['user_register_time']
data['time_diff4'] = data['request_time_min'] - data['user_least_login_time']
data['time_diff5'] = data['request_time_max'] - data['user_least_login_time']

data['time_range'] = data['request_time_max'] - data['request_time_min']
data['active_means'] = data['request_request_model_id_count'] / data['time_range']

cat_cols = ['request_model_id', 'request_ip',
            'request_device_type', 'request_app_version',
            'request_app_channel']
for f in cat_cols:
    data[f + '_base_count'] = data[f].map(data[f].value_counts())

for col in cat_cols:
    data[col] = data[col].fillna(' ')
    lb = LabelEncoder()
    data[col] = lb.fit_transform(data[col])

# 训练集测试集划分
train = data[~data['request_label'].isna()].reset_index(drop=True)
test = data[data['user'].isin(user_list2)].reset_index(drop=True)

features = [i for i in train.columns if i not in ['user', 'user_status', 'request_label', 'request_target', ]]
y = train['request_label']
print(len(features))

def train_model(X_train, X_test, features, y, save_model=False):
    """
    训练lgb模型
    """
    feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
    KF = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'n_jobs': -1,
        'learning_rate': 0.05,
        'num_leaves': 2 ** 6,
        'max_depth': 8,
        'tree_learner': 'serial',
        'colsample_bytree': 0.8,
        'subsample_freq': 1,
        'subsample': 0.8,
        'num_boost_round': 5000,
        'max_bin': 255,
        'verbose': -1,
        'seed': 2021,
        'bagging_seed': 2021,
        'feature_fraction_seed': 2021,
        'early_stopping_rounds': 100,
    }
    oof_lgb = np.zeros(len(X_train))
    predictions_lgb = np.zeros((len(X_test)))

    for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
        trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y.iloc[val_idx])
        num_round = 10000
        clf = lgb.train(
            params,
            trn_data,
            num_round,
            valid_sets=[trn_data, val_data],
            verbose_eval=100,
            early_stopping_rounds=50,
        )

        oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions_lgb[:] += clf.predict(X_test[features], num_iteration=clf.best_iteration) / 5
        feat_imp_df['imp'] += clf.feature_importance() / 5
        if save_model:
            clf.save_model(f'model_{fold_}.txt')

    print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
    print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))

    return feat_imp_df, oof_lgb, predictions_lgb


feat_imp_df, oof_lgb, predictions_lgb = train_model(train, test, features, y)

# 取user概率最大值作为预测结果
train['pred'] = oof_lgb
train['pred_max'] = train.groupby('user')['pred'].transform('max')
train['label'] = train['pred_max'].apply(lambda x: 1 if x > 0.5 else 0)
t = train.drop_duplicates('user')

print("AUC score: {}".format(roc_auc_score(t['request_label'], t['pred_max'])))
print("F1 score: {}".format(f1_score(t['request_label'], t['label'])))
print("Precision score: {}".format(precision_score(t['request_label'], t['label'])))
print("Recall score: {}".format(recall_score(t['request_label'], t['label'])))


# 输出结果文件
test['pred'] = predictions_lgb
test['pred_max'] = test.groupby('user')['pred'].transform('max')
test['label'] = test['pred_max'].apply(lambda x: 1 if x > 0.5 else 0)
t = test.drop_duplicates('user')
t[['user', 'label']].to_csv('./submission/sub_requests_final.csv', header=None, index=False)
t['label'].sum()



