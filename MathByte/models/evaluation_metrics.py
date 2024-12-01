import numpy as np
import tensorflow as tf
import keras.backend as K
import logging
import tensorflow_addons as tfa


np.random.seed(3407)
tf.random.set_seed(3407)

# how to convert tensor to numpy
# using tf.py_function
def accuracy_1k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: accuracy_k(y_true, y_pred, 0), inp=[y_true, y_pred], Tout=tf.float32)

def accuracy_2k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: accuracy_k(y_true, y_pred, 1), inp=[y_true, y_pred], Tout=tf.float32)

def accuracy_3k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: accuracy_k(y_true, y_pred, 2), inp=[y_true, y_pred], Tout=tf.float32)

def accuracy_5k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: accuracy_k(y_true, y_pred, 4), inp=[y_true, y_pred], Tout=tf.float32)

def precision_1k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: precision_k(y_true, y_pred, 0), inp=[y_true, y_pred], Tout=tf.float32)


def precision_2k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: precision_k(y_true, y_pred, 1), inp=[y_true, y_pred], Tout=tf.float32)


def precision_3k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: precision_k(y_true, y_pred, 2), inp=[y_true, y_pred], Tout=tf.float32)


def precision_5k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: precision_k(y_true, y_pred, 4), inp=[y_true, y_pred], Tout=tf.float32)


def recall_1k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: recall_k(y_true, y_pred, 0), inp=[y_true, y_pred], Tout=tf.float32)


def recall_2k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: recall_k(y_true, y_pred, 1), inp=[y_true, y_pred], Tout=tf.float32)


def recall_3k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: recall_k(y_true, y_pred, 2), inp=[y_true, y_pred], Tout=tf.float32)


def recall_5k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: recall_k(y_true, y_pred, 4), inp=[y_true, y_pred], Tout=tf.float32)


def F1_1k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: F1_k(y_true, y_pred, 0), inp=[y_true, y_pred], Tout=tf.float32)


def F1_2k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: F1_k(y_true, y_pred, 1), inp=[y_true, y_pred], Tout=tf.float32)


def F1_3k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: F1_k(y_true, y_pred, 2), inp=[y_true, y_pred], Tout=tf.float32)


def F1_5k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: F1_k(y_true, y_pred, 4), inp=[y_true, y_pred], Tout=tf.float32)


def Ndcg_1k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: Ndcg_k(y_true, y_pred, 0), inp=[y_true, y_pred], Tout=tf.float32)


def Ndcg_3k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: Ndcg_k(y_true, y_pred, 2), inp=[y_true, y_pred], Tout=tf.float32)


def Ndcg_5k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: Ndcg_k(y_true, y_pred, 4), inp=[y_true, y_pred], Tout=tf.float32)


def hamming_loss1(y_true, y_pred):
    metric = tfa.metrics.HammingLoss(mode='multilabel')
    metric.update_state(y_true, y_pred)
    return metric.result()


def hamming_loss_k(y_true, y_pred):
    return tf.py_function(func=lambda y_true, y_pred: hamming_loss1(y_true, y_pred), inp=[y_true, y_pred], Tout=tf.float32)


def basic_metrics():
    return [precision_1k, precision_2k, precision_3k, precision_5k, recall_1k, recall_2k, recall_3k, recall_5k, F1_1k,
            F1_2k, F1_3k, F1_5k, accuracy_1k, accuracy_2k, accuracy_3k, accuracy_5k, hamming_loss_k]




def lcm_metrics(num_classes, alpha):

    def lcm_loss(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        label_sim_dist = y_pred[:, num_classes:]
        simulated_y_true = K.softmax(label_sim_dist+alpha*y_true)
        loss1 = - \
            K.categorical_crossentropy(simulated_y_true, simulated_y_true)
        loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)
        return loss1+loss2

    def lcm_accuracy_1k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return accuracy_1k(y_true, pred_probs)

    def lcm_accuracy_2k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return accuracy_2k(y_true, pred_probs)

    def lcm_accuracy_3k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return accuracy_3k(y_true, pred_probs)

    def lcm_accuracy_5k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return accuracy_5k(y_true, pred_probs)

    def lcm_precision_1k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return precision_1k(y_true, pred_probs)

    def lcm_precision_2k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return precision_2k(y_true, pred_probs)

    def lcm_precision_3k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return precision_3k(y_true, pred_probs)

    def lcm_precision_5k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return precision_5k(y_true, pred_probs)

    def lcm_recall_1k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return recall_1k(y_true, pred_probs)

    def lcm_recall_2k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return recall_2k(y_true, pred_probs)

    def lcm_recall_3k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return recall_3k(y_true, pred_probs)

    def lcm_recall_5k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return recall_5k(y_true, pred_probs)

    def lcm_f1_1k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return F1_1k(y_true, pred_probs)

    def lcm_f1_2k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return F1_2k(y_true, pred_probs)

    def lcm_f1_3k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return F1_3k(y_true, pred_probs)

    def lcm_f1_5k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return F1_5k(y_true, pred_probs)

    def lcm_hamming_loss_k(y_true, y_pred):
        pred_probs = y_pred[:, :num_classes]
        return hamming_loss_k(y_true, pred_probs)

    return lcm_loss, [lcm_precision_1k, lcm_precision_2k, lcm_precision_3k, lcm_precision_5k, lcm_recall_1k,
                      lcm_recall_2k, lcm_recall_3k, lcm_recall_5k, lcm_f1_1k, lcm_f1_2k, lcm_f1_3k, lcm_f1_5k,
                      lcm_accuracy_1k, lcm_accuracy_2k, lcm_accuracy_3k, lcm_accuracy_5k, lcm_hamming_loss_k]


def my_evaluator(y_true, y_pred):
    logging.info("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " %
                 (precision_1k(y_true, y_pred), precision_3k(y_true, y_pred), precision_5k(y_true, y_pred)))
    logging.info("recall@1 : %.4f , recall@3 : %.4f , recall@5 : %.4f " %
                 (recall_1k(y_true, y_pred), recall_3k(y_true, y_pred), recall_5k(y_true, y_pred)))
    logging.info("f1@1 : %.4f ,f1@3 : %.4f , f1@5 : %.4f " %
                 (F1_1k(y_true, y_pred), F1_3k(y_true, y_pred), F1_5k(y_true, y_pred)))
    logging.info("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " %
                 (Ndcg_1k(y_true, y_pred), Ndcg_3k(y_true, y_pred), Ndcg_5k(y_true, y_pred)))


def precision_k(y_true, y_pred, k):
    '''
    Precision@k
    @param: y_true: shape=(None, None)
    @param: y_pred: shape=(None, num_classes)
    @return: (5, 1)
    '''
    top_k = 5
    # CAST YOUR TENSOR & CONVERT IT TO NUMPY ARRAY
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    p = np.zeros((top_k, 1))
    rank_mat = np.argsort(y_pred)
    backup = np.copy(y_pred)
    for m in range(top_k):
        y_pred = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            y_pred[i][rank_mat[i, :-(m + 1)]] = 0
        y_pred = np.ceil(y_pred)
        mat = np.multiply(y_pred, y_true)
        num = np.sum(mat, axis=1)
        p[m] = np.mean(num / (m + 1))
    return np.around(p, decimals=4)[k]


def recall_k(y_true, y_pred, k):
    '''
    Recall@k
    @param: y_true: shape=(None, None)
    @param: y_pred: shape=(None, num_classes)
    @return: (5, 1)
    '''
    top_k = 5
    # CAST YOUR TENSOR & CONVERT IT TO NUMPY ARRAY
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    p = np.zeros((top_k, 1))
    rank_mat = np.argsort(y_pred)
    backup = np.copy(y_pred)
    all_num = np.sum(y_true, axis=1)  # 所有的正类
    for m in range(top_k):
        y_pred = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            y_pred[i][rank_mat[i, :-(m + 1)]] = 0
        y_pred = np.ceil(y_pred)
        mat = np.multiply(y_pred, y_true)
        num = np.sum(mat, axis=1)
        p[m] = np.mean(num / all_num)
    return np.around(p, decimals=4)[k]


def F1_k(y_true, y_pred, k):
    '''
    F1@k
    @param: y_true: shape=(None, None)
    @param: y_pred: shape=(None, num_classes)
    @return: (5, 1)
    '''
    p_k = precision_k(y_true, y_pred, k)
    r_k = recall_k(y_true, y_pred, k)
    if p_k + r_k == 0:
        return 0.0
    return (2 * p_k * r_k) / (p_k + r_k)


def Ndcg_k(y_true, y_pred, k):
    '''
    自定义评价指标
    @param: y_true: shape=(None, None)
    @param: y_pred: shape=(None, num_classes)
    @return: (5, 1)
    '''
    top_k = 5
    # CAST YOUR TENSOR & CONVERT IT TO NUMPY ARRAY
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    res = np.zeros((top_k, 1))
    rank_mat = np.argsort(y_pred)
    label_count = np.sum(y_true, axis=1)

    for m in range(top_k):
        y_mat = np.copy(y_true)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m + 1)]] = 0
            for j in range(m + 1):
                y_mat[i][rank_mat[i, -(j + 1)]] /= np.log(j + 1 + 1)

        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m + 1)
        ndcg = np.mean(dcg / factor)
        res[m] = ndcg
    return np.around(res, decimals=4)[k]


def get_factor(label_count, k):
    res = []
    for i in range(len(label_count)):
        n = int(min(label_count[i], k))
        f = 0.0
        for j in range(1, n+1):
            f += 1/np.log(j+1)
        res.append(f)
    return np.array(res)


def accuracy_k(y_true, y_pred, k):
    """
    Computes the top-k accuracy. The score is calculated by considering if the true label is within the top-k scores.

    :param y_true: 2D list or numpy array of shape [n_samples, n_labels]
    :param y_score: 2D list or numpy array of shape [n_samples, n_labels], confidence scores for each label
    :param k: int, the number of top elements to look at for computing the accuracy
    :return: float, top-k accuracy
    """

    top_k = 5
    # CAST YOUR TENSOR & CONVERT IT TO NUMPY ARRAY
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    correct_preds = np.zeros(top_k)  # 用于存储每个k值的正确预测数
    rank_mat = np.argsort(y_pred)
    backup = np.copy(y_pred)
    for m in range(top_k):
        y_pred = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            y_pred[i][rank_mat[i, :-(m + 1)]] = 0
        y_pred = np.ceil(y_pred)
        correct_preds[m] = np.sum(np.any(y_pred * y_true, axis=1))
    accuracies = correct_preds / y_true.shape[0] # 正确预测的总数 / 总样本数
    return np.around(accuracies, decimals=4)[k]








