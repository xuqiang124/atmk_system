import keras
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LayerNormalization, Dropout
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel


np.random.seed(3407)
tf.random.set_seed(3407)

class Classifier(object):
    """
    分类器
    """
    @classmethod
    def build(self, config, use_att=False, label_emb_matrix=None, metrics=None):

        # 共有变量
        maxlen = config.maxlen
        hidden_size = config.hidden_size
        num_classes_list = config.num_classes_list


        def _attention(input_x, label_emb_matrix, name=""):
            """
            Attention Layer.
            Args:
                input_x: [batch_size,max_length, hidden_size]
                label_emb_matrix: the embedding matrix of i-th labels [batch_size,num_classes,hidden_size]
                name: Scope name.
            Returns:
                attention_out: [batch_size, hidden_size]
            """
            # 使用 tf.matmul 计算注意力权重 (None, num_classes, max_length)
            attention_weight = tf.matmul(label_emb_matrix, tf.transpose(input_x, perm=[0, 2, 1]), name=f"{name}_weight")
            # 激活权重
            attention_weight_probs =  tf.sigmoid(attention_weight)

            # 施加注意力权重于原始文本表示
            weighted_text_representation = tf.matmul(attention_weight_probs,input_x)
            # weighted_text_representation: (batch_size, num_classes, hidden_size)

            # 通过对加权文本表示进行平均来得到最终表示
            attention_out = tf.reduce_mean(weighted_text_representation, axis=1, name=f"{name}_context")
            # final_representation: (batch_size, hidden_size)

            # 添加归一化层和 dropout
            attention_out = LayerNormalization()(attention_out)
            attention_out = Dropout(0.1)(attention_out)  # dropout rate 可以通过 config 调整
            return attention_weight_probs, attention_out

        def _local_layer(input_x,input_att_weight, num_classes, name=""):
            """
            Local Layer.
            Args:
                input_x: [batch_size, hidden_size]
                input_att_weight: [batch_size, num_classes, sequence_length]
                num_classes: Number of classes
                name: Scope name.
            Returns:
                predict_transmit: [batch_size, sequence_length]
            """
            # 得到概率得分
           # 预测概率，Dense层的name加上层名前缀  [batch_size, num_classes]
            scores = Dense(num_classes, activation='sigmoid', name=f"{name}_scores")(input_x)
            # 计算 softmax 概率，tf.expand_dims 保持维度
            visual = tf.multiply(input_att_weight, tf.expand_dims(scores, -1)) # [batch_size, num_classes, sequence_length]
            visual = tf.nn.softmax(visual) # [batch_size, num_classes, sequence_length]
            visual = tf.reduce_mean(visual, axis=1, name="visual")# [batch_size, sequence_length]
            return visual
        
        # 输入层
        input_ids = Input(shape=(maxlen,), dtype='int32', name='input_ids')
        attention_mask = Input(shape=(maxlen,), dtype='int32', name='attention_mask')

        # RoBERTa 编码器层
        roberta = TFBertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        # # 冻结roberta前面的几个 Transformer 层，比如前6层
        label_trainable = False
        if use_att:
            label_trainable = True
            for i in range(6):  # 这里冻结了前6层
                roberta.bert.encoder.layer[i].trainable = False
        roberta_output = roberta(input_ids, attention_mask=attention_mask)[0]  # [0]表示最后一层隐藏状态 #shape (None, max_len, hidden_size)
        
        # 计算所有标签数目
        hierarchy_levels=len(num_classes_list)
        count = sum(num_classes_list[i] for i in range(hierarchy_levels))
        label_input = Input(shape=(count,), name='label_input') # 标签


        # 标签预训练
        if label_emb_matrix is None:
            # shape=(None, all_num_classes, hidden_size)
            label_emb = Embedding(
                count, hidden_size, input_length=count, name='label_emb')(label_input)
        else:
            label_emb = Embedding(count, hidden_size, input_length=count, weights=[
                label_emb_matrix], trainable=label_trainable, name='label_emb')(label_input)

       
        if use_att:  # 标签注意力
            idx = 0
            for i in range(hierarchy_levels):
                level_label_emb = label_emb[:, idx:idx+num_classes_list[i], :]
                idx+=num_classes_list[i]
                # attention_out shape=(None, hidden_size)
                # attention_weight shape=(None, num_classes, sequence_length)
                attention_weight, attention_out = _attention(roberta_output, level_label_emb, str(i)+"_attention_layer_") # shape=(None, hidden_size)
                roberta_output_pool = roberta_output[:, 0, :] # 使用 [CLS] token 的隐藏状态作为分类输入 #shape (None, hidden_size)
                weight1 = Dense(1, activation='sigmoid', input_shape=(hidden_size,))(roberta_output_pool)
                weight2 = Dense(1, activation='sigmoid', input_shape=(hidden_size,))(attention_out)
                weight1 = weight1 / (weight1 + weight2)
                weight2 = 1 - weight1
                doc = weight1 * roberta_output_pool + weight2 * attention_out # shape=(None, hidden_size)
                
                if (i != (hierarchy_levels-1)):
                    # get_transmit
                    local_transmit = _local_layer(doc,attention_weight, num_classes_list[i], str(i) + "_local_layer_") # [None, max_length]
                    roberta_output = tf.multiply(roberta_output, tf.expand_dims(local_transmit, -1))
                else:
                    roberta_output = doc
        else:
            roberta_output = roberta_output[:, 0, :]
        pred_probs = Dense(num_classes_list[-1], activation='sigmoid',
                           name='pred_probs')(roberta_output)
       
        model = Model(inputs=[input_ids,attention_mask, label_input], outputs=pred_probs)
        
        optimizer = Adam(learning_rate=2e-5)
        # 每一批次评估一次
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer, metrics=metrics)  # 自定义评价函数

        model._get_distribution_strategy = lambda: None  # fix bug for 2.1 tensorboard
        print(model.summary())
        return model, roberta_output, label_emb[:, -num_classes_list[-1]:, :]
