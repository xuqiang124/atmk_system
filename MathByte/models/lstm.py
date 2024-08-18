import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Bidirectional, Dropout
from keras.layers import Flatten, Concatenate, Permute, Lambda, Dot
import keras.backend as K
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

np.random.seed(3407)
tf.random.set_seed(3407)

class Classifier(object):
    """
    分类器
    """
    @classmethod
    def build(self, config, embedding_matrix=None, use_att=False, label_emb_matrix=None, metrics=None):

        maxlen = config.maxlen
        vocab_size = config.vocab_size
        wvdim = config.emb_size
        hidden_size = config.hidden_size
        num_classes_list = config.num_classes_list

        def _attention(input_x, label_emb_matrix, sequence_length,num_classes, name=""):
            """
            Attention Layer.
            Args:
                input_x: [batch_size, sequence_length(maxlen), lstm_hidden_size * 2]
                label_emb_matrix: the embedding matrix of i-th labels [None,num_classes,wvdim]
                sequence_length: length of sequence
                num_classes: the labels of i-th level
                name: Scope name.
            Returns:
                attention_out: [batch_size, lstm_hidden_size * 2]
            """

            # 计算注意力权重 attention_weight attention_weight_probs shape=(None, num_classes, sequence_length)
            attention_weight = Lambda(lambda x: K.batch_dot(
                *x))([label_emb_matrix, Permute((2, 1))(input_x)])  # 计算注意力权重
            attention_weight_probs = Dense(sequence_length, activation='sigmoid', name=name+'att_weight')(attention_weight)
            # 注意力输出 shape=(None, num_classes, hidden_size*2)
            attention_out_matrix = Lambda(lambda x: K.batch_dot(*x))([attention_weight_probs, input_x])
            # 平均值到 r shape=(None, hidden_size*2)
            attention_out = Lambda(lambda x: K.sum(x, 1) / num_classes, name=name+"att_context")(attention_out_matrix)
            # attention_out = tf.reduce_mean(attention_out_matrix, axis=1)
            return attention_out_matrix, attention_out

        def _local_layer(input_x, num_classes, sequence_length, origin_input, name=""):
            """
            Local Layer.
            Args:
                input_x: [batch_size, hidden_size*2]
                num_classes: Number of classes.
                sequence_length: length of sequence
                origin_input: [batch_size, sequence_length(maxlen), lstm_hidden_size * 2]
                name: Scope name.
            Returns:
                logits: [batch_size, num_classes]
                scores: [batch_size, num_classes]
                visual: [batch_size, sequence_length]
            """
            # 得到概率得分
            # scores shape=(None, num_classes)
            scores = Dense(num_classes, activation='sigmoid')(input_x)
            visual = tf.nn.softmax(scores)  # shape=(None, num_classes)
            visual = tf.expand_dims(visual, -1)  # shape=(None, num_classes, 1)
            


            # 预测结果嵌入矩阵
            predict_att_emb = Dense(hidden_size * 2, activation='tanh', name=name+'predict_att_emb')(
                visual)  # shape=(None, num_classes, hidden_size * 2)


            # 计算预测结果注意力权重 attention_weight attention_weight_probs shape=(None, num_classes, sequence_length)
            label_attention_weight = Lambda(lambda x: K.batch_dot(
                *x))([predict_att_emb, Permute((2, 1))(origin_input)])  # 计算注意力权重
            label_attention_weight_probs = Dense(sequence_length, activation='sigmoid')(label_attention_weight)

            # 在类别维度进行平均，shape=(None, time_steps)
            predict_transmit = tf.reduce_mean(label_attention_weight_probs, axis=1, name=name+"predict_transmit")

            return predict_transmit

        text_input = Input(shape=(maxlen,), name='text_input')

        if embedding_matrix is None:
            input_emb = Embedding(
                vocab_size, wvdim, input_length=maxlen, name='text_emb')(text_input)  # (V,wvdim)
        else:
            input_emb = Embedding(vocab_size, wvdim, input_length=maxlen, weights=[
                                  embedding_matrix], trainable=False, name='text_emb')(text_input)  # (V,wvdim)
        # NOTE 使用注意力则返回全部step，否则返回最后step
        # shape=(None, maxlen, hidden_size * 2) or shape=(None, hidden_size * 2)
        lstm_output = Bidirectional(LSTM(hidden_size, return_sequences=use_att), name='BiLSTM')(
            input_emb)
        # lstm_output = Dropout(rate=0.2)(lstm_output_temp)

        hierarchy_levels = len(num_classes_list)

        count = sum(num_classes_list[i] for i in range(hierarchy_levels))
        # 标签
        label_input = Input(shape=(count,), name='label_input')
        # 标签预训练
        if label_emb_matrix is None:
            # shape=(None, num_classes, wvdim)
            label_emb = Embedding(
                num_classes_list[-1], wvdim, input_length=num_classes_list[-1], name='label_emb')(label_input)
        else:
            label_emb = Embedding(num_classes_list[-1], wvdim, input_length=num_classes_list[-1], weights=[
                label_emb_matrix], trainable=False, name='label_emb')(label_input)



        if use_att:  # 标签注意力
            sequence_length = K.int_shape(lstm_output)[1]
            idx = 0
            for i in range(hierarchy_levels):
                level_label_emb = label_emb[:, idx:idx+num_classes_list[i], :]
                idx+=num_classes_list[i]
                # 计算标签矩阵 label_emb_matrix shape=(None, level_num_classes, hidden_size*2)
                label_emb_matrix = Dense(hidden_size * 2, activation='tanh', name=str(i)+'_level_label_emb')(level_label_emb)
                attention_out_matrix, attention_out = _attention(lstm_output, label_emb_matrix,sequence_length, num_classes_list[i], str(i)+"_attention_layer_")
                if (i != (hierarchy_levels-1)):
                    lstm_output_pool = K.mean(lstm_output, axis=1)  # [None, lstm_hidden_size * 2]
                    # local_input shape=(None, hidden_size*4)
                    local_input = tf.concat([lstm_output_pool, attention_out], axis=1)
                    # local_fc_out shape=(None, hidden_size*2)
                    local_fc_out = Dense(hidden_size * 2, activation='relu')(local_input)
                    #add dropout
                    # local_fc_out_dropout = Dropout(rate=0.2)(local_fc_out)
                    # get_transmit
                    local_transmit = _local_layer(local_fc_out, num_classes_list[i], sequence_length, lstm_output, str(i) + "_local_layer_")
                    lstm_output = tf.multiply(lstm_output, tf.expand_dims(local_transmit, -1))
                else:
                    lstm_output = attention_out

        pred_probs = Dense(num_classes_list[-1], activation='sigmoid',
                           name='pred_probs')(lstm_output)

        model = Model(inputs=[text_input, label_input], outputs=pred_probs)
        # 每一批次评估一次
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam', metrics=metrics)  # 自定义评价函数

        model._get_distribution_strategy = lambda: None  # fix bug for 2.1 tensorboard
        print(model.summary())
        return model, lstm_output, label_emb[:, -num_classes_list[-1]:, :]
