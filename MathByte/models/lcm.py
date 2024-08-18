import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Flatten, Dropout, Concatenate, Lambda, Multiply, Reshape, Dot, Bidirectional


class LabelConfusionModel(object):
    """
    分类器
    """
    @classmethod
    def build(self, config, basic_model, text_h_state, label_emb, loss, metrics):
        hidden_size = config.hidden_size
        num_classes = config.num_classes_list[-1]

        # 乘2是因为text用的BiLSTM
        label_lcm_emb = Dense(hidden_size*2, activation='tanh',
                              name='label_lcm_emb')(label_emb)  # shape=(None, num_classes, hidden_size*2)
        # similarity part:
        # (num_classes,hidden_size*2) dot (hidden_size*2,1) --> (num_classes,1)
        # text_h_state = basic_model.layers[-1].input  # 取text最后一层的输入
        doc_product = Dot(axes=(2, 1))(
            [label_lcm_emb, text_h_state])  # shape=(None, num_classes)
        # 标签模拟分布
        label_sim_dict = Dense(
            num_classes, activation='softmax', name='label_sim_dict')(doc_product)
        # concat output:
        # shape=(None, text_d+label_d)
        concat_output = Concatenate()([basic_model.outputs[0], label_sim_dict])

        # compile；
        model = Model(
            inputs=basic_model.inputs, outputs=concat_output)
        model.compile(loss=loss, optimizer='Adam', metrics=metrics)
        model._get_distribution_strategy = lambda: None
        print(model.summary())
        return model
