import numpy as np


def dcg_k(score_label, k):
    dcg, i = 0.0, 0
    for s in score_label:
        if i < k:
            dcg += (2 ** s[1] - 1) / np.log2(2 + i)
            i += 1
    return dcg


def ndcg_k(y_hat, y, k):
    score_label = np.stack([y_hat, y], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d: d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d: d[1], reverse=True)
    norm, i = 0.0, 0
    for s in score_label_:
        if i < k:
            norm += (2 ** s[1] - 1) / np.log2(2 + i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm


def call_ndcg(y_hat, y):
    ndcg_sum, num = 0, 0
    y_hat, y = y_hat.T, y.T
    n_users = y.shape[0]

    for i in range(n_users):
        y_hat_i = y_hat[i][np.where(y[i])]
        y_i = y[i][np.where(y[i])]

        if y_i.shape[0] < 2:
            continue

        ndcg_sum += ndcg_k(y_hat_i, y_i, y_i.shape[0])  # user-wise calculation
        num += 1

    return ndcg_sum / num
