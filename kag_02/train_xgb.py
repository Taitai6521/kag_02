import pandas as pd
import numpy as np
import pickle


from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc

import xgboost as xgb
from test import load_train_data, load_test_data

logger = getLogger(__name__)


DIR = 'result_tmp/tmp'
SAMPLE_SUBMIT_FILE = 'input/sample_submission.csv'




def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr,tpr) - 1
    return g

def gini_xgb(pred,y):
    y = y.get_label()
    return 'gini', gini(y, pred)



if __name__ == '__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)



    logger.info('start')

    df = load_train_data()



    x_train = df.drop('target', axis=1)
    y_train = df['target'].values


    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end {}'.format(x_train.shape))

    #cross validation para01 start
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # 初期の値です。
    # all_params = {'C': [10**i for i in range(-1,2)],
    #               'fit_intercept': [True, False],
    #               'penalty': ['l2', 'l1'],
    #               'random_state': [0],
    #               'seed': [0]
    #
    #               }

    all_params = {'max_depth': [3, 5, 7],
                  'learning_rate': [0.1],
                  'min_child_weight': [3, 5, 100],
                  'n_estimaters': [100000],
                  'colsample_bytree': [0.8, 0.9],
                  'colsample_bylevel': [0.8, 0.9],
                  'reg-alpha':[0, 0.1],
                  'max_delta_step': [0.1],





                  'seed': [0]}



    min_score = 100
    min_params = None



    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))




        list_gini_score = []
        list_logloss_score = []


        for train_idx, valid_idx in cv.split(x_train, y_train):
            trn_x = x_train.iloc[train_idx, :]
            val_x = x_train.iloc[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]
            #logisticの変更点,liblinearに変更
            clf = xgb.sklearn.XGBClassifier(**params, solver='liblinear')
            clf.fit(trn_x,
                    trn_y,
                    eval_set=[(val_x, val_y)],
                    early_stopping_rounds=10,
                    eval_metric = gini_xgb
                    )


            #log_los,auc=start
            pred = clf.predict_proba(val_x, ntree_limit = clf.best_ntree_limit)[:, 1]




            sc_logloss = log_loss(val_y, pred)
            sc_gini = - gini(val_y, pred)

            list_logloss_score.append(sc_logloss)
            list_gini_score.append(sc_gini)
            list_best_iterations.append(clf.best_iteration)

            logger.debug(' logloss: {}, auc: {}'.format(sc_logloss, sc_gini))

            #log_los,auc=finish

            #成功してからの追加
            break
        params['n_estimators'] = int(np.mean(list_best_iterations))
        sc_logloss = np.mean(list_logloss_score)
        sc_gini = -np.mean(list_gini_score)

        #
        if min_score > sc_gini:
            min_score = sc_gini
            min_params = params
        logger.info('logloss: {}, auc: {}'.format(sc_logloss, sc_gini))
        logger.info('current min score: {}, min_params: {}'.format(min_score, min_params))


    logger.info('minimum_params　: {}'.format(min_params))
    logger.info('minimum_auc: {}'.format(min_score))



    clf = xgb.sklearn.XGBClassifier(**min_params, solver='liblinear')
    clf.fit(x_train,
            y_train,
    #         )

    with open(DIR + 'model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)


    #
    # #cross finish
    # logger.info('train end')
    #
    #
    # ##train　finish。
    #
    #
    #
    # df = load_test_data()
    # x_test = df[use_cols].sort_values('id')
    #
    #
    # logger.info('test data load end {}'.format(x_test.shape))
    #
    # pred_test = clf.predict_proba(x_test)
    #
    # #test finish。
    #
    #
    # df_submit =pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    # df_submit['target'] = pred_test
    #


    df_submit.to_csv(DIR + 'submit.csv', index=False)

    logger.info('end')
    #sample finish

