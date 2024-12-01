import numpy as np
import os
import keras
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger

from .roberta import Classifier
from .lcm import LabelConfusionModel
from .evaluation_metrics import basic_metrics, lcm_metrics

np.random.seed(3407)

class LHABSModel:

    def __init__(self, config, label_emb_matrix=None, use_att=False, use_lcm=False, log_dir=None):
        self.epochs = config.epochs
        self.alpha = config.alpha
        self.num_classes_list = config.num_classes_list
        self.batch_size = config.batch_size
        self.use_att = use_att
        self.use_lcm = use_lcm
        # self.model_filepath = os.path.join(
        #     log_dir, "model")
        self.model_filepath = os.path.join(
            log_dir, "model", self.__get_saved_model_name())

        self.basic_model, hid, label_emb = Classifier.build(
            config, use_att, label_emb_matrix, basic_metrics())
        es_monitor = "val_loss"
        # mc_monitor = "val_precision_1k"
        patience = config.l_patience
        if (use_att == False) & (use_lcm == False):
            patience = config.b_patience  # basic 模型做较大设置
        print(patience, "patience")

        if use_lcm:
            loss, metrics = lcm_metrics(self.num_classes_list[-1], self.alpha)
            self.model = LabelConfusionModel.build(
                config, self.basic_model, hid, label_emb, loss, metrics)
            # mc_monitor = "val_lcm_precision_1k"
        # 设置训练过程中的回调函数
        tb = TensorBoard(log_dir=os.path.join(log_dir, "fit"))
        # 设置 early stop
        es = EarlyStopping(monitor=es_monitor, mode='min',
                           verbose=1, patience=patience, min_delta=0.0001)
        # 保存 val_loss 最小时的model
        # mc = ModelCheckpoint(filepath=os.path.join(self.model_filepath,"LABS_epoch_{epoch:02d}.h5"), monitor=es_monitor,
        #                      mode='min', verbose=1, save_best_only=False, save_freq=30)

        mc = ModelCheckpoint(self.model_filepath, monitor=es_monitor,
                             mode='min', verbose=1, save_best_only=True,save_weights_only=False)
        # 保存训练过程数据到csv文件
        logger = CSVLogger(os.path.join(log_dir, "training.csv"))

        # 添加学习率到回调函数中
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
        self.callbacks = [tb,es, mc, logger]

    def train(self, X_train,X_train_mask,y_train, L_train):
        model = self.model if self.use_lcm else self.basic_model
        model.fit([X_train,X_train_mask,L_train], y_train,
                  batch_size=self.batch_size, verbose=1, epochs=self.epochs, validation_split=0, callbacks=self.callbacks)
        # X_train_1, X_val, L_train_1 , L_val, y_train_1, y_val = train_test_split(X_train, L_train, y_train, test_size=0.2, random_state=3407)
        # model.fit([X_train_1, L_train_1], y_train_1,
        #           batch_size=self.batch_size, verbose=1, epochs=self.epochs, validation_data=([X_val, L_val], y_val), callbacks=self.callbacks)

    def train_and_val(self, X_train, X_train_mask, y_train, L_train, X_val, X_val_mask, y_val, L_val):
        model = self.model if self.use_lcm else self.basic_model
        model.fit([X_train, X_train_mask, L_train], y_train,
                  batch_size=self.batch_size, verbose=1, epochs=self.epochs, validation_data=([X_val, X_val_mask, L_val], y_val), callbacks=self.callbacks)

    def validate(self, X_test, X_test_mask, y_test, L_test):
        # load the saved model
        saved_model = self.model if self.use_lcm else self.basic_model
        saved_model.load_weights(self.model_filepath)

        # 现在你可以继续使用 saved_model 进行评估或微调
        # evaluate the model
        result = saved_model.evaluate([X_test, X_test_mask, L_test], y_test, verbose=1)
        print(f"Model: {self.model_filepath}, Test Result: {result}")
        return result
    
    def get_load_model(self,model_filepath):
        saved_model = self.model if self.use_lcm else self.basic_model
        saved_model.load_weights(model_filepath)
        return saved_model

    def __get_saved_model_name(self, ):
        '''
        {epoch:02d}-{val_lcm_precision_1k:.2f}
        '''
        if self.use_lcm and self.use_att:
            return "checkpoint_lhabs.h5"
        elif self.use_lcm:
            return "checkpoint_lbs.h5"
        elif self.use_att:
            return "checkpoint_lhab.h5"
        else:
            return "checkpoint_b.h5"
    
   
