import pickle
from gensim.models import Word2Vec, KeyedVectors

from formula_embedding.tangent_cft_back_end import TangentCFTBackEnd
from math_questions.const import WORD_MODEL_PATH, CHAR_MODEL_PATH, BAIDU_MODEL_PATH

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Embeddings:
    def __init__(self, cache_file_pickle=None, cache_embedding_file=None) -> None:
        self.cache_file_pickle = cache_file_pickle
        self.cache_embedding_file = cache_embedding_file

    def read_text_vec(self, type_id, query, version='atmk'):
        '''
        获取字or词的向量
        :param type_id: 向量类型 char | word
        :param query: 字符或词语
        :param version: 使用的中文词向量版本 atmk | baidu
        :return: 向量
        '''
        if version == 'baidu':
            model_file_path = BAIDU_MODEL_PATH
            # 加载已经预训练好的词向量
            math_word2vec_model = KeyedVectors.load_word2vec_format(model_file_path, binary=False)
            # text_vec = math_word2vec_model.wv[query]
            text_vec = math_word2vec_model.get_vector(query)
            return text_vec
        else:
            model_file_path = 'D:/GithubProject/atmk_system/file_data/math_text_userdict.model'
            if type_id == 'word':
                model_file_path = 'D:/GithubProject/atmk_system/file_data/math_text_userdict.model'
            math_word2vec_model = Word2Vec.load(model_file_path)
            text_vec = math_word2vec_model.wv[query]
            return text_vec

    def read_formula_vec(self, query_formula, version='atmk'):
        '''
        获取公式向量
        :param query_formula: 公式
        :param version: 使用的词向量版本 atmk | wiki
        :return: 公式向量
        '''
        model_file_path = 'file_data/da-20k/slt_model'  # Model file path
        map_file_path = 'file_data/da-20k/slt_encoder.tsv'
        if version == 'wiki':
            model_file_path = 'file_data/wiki-590k/slt_model'
            map_file_path = 'file_data/wiki-590k/slt_encoder.tsv'

        key = 'hello_world'
        query_formulas = [{
            'key': key,
            'content': query_formula
        }]
        system = TangentCFTBackEnd(
            config_file=None, data_set=None, query_formulas=query_formulas)
        system.load_model(map_file_path=map_file_path,
                          model_file_path=model_file_path)
        formula_vec = system.get_collection_query_vectors()[key]
        return formula_vec

    def batch_read_text_vec(self, query_text, token_type, version):
        '''批量读取字or词向量'''
        math_word2vec_model = None
        if version == 'baidu':
            model_file_path = BAIDU_MODEL_PATH
            math_word2vec_model = KeyedVectors.load_word2vec_format(
                model_file_path, binary=False)
        else:
            model_file_path = CHAR_MODEL_PATH
            if token_type == 'word':
                model_file_path = '../file_data/aaai/new/formula/math_text_word.model'
            math_word2vec_model = Word2Vec.load(model_file_path)
        ret = {}
        for k, v in query_text.items():
            try:
                ret[k] = math_word2vec_model.wv[v]
            except Exception:
                ret[k] = None
        return ret

    def batch_read_formula_vec(self, query_formula, version):
        '''批量读取公式向量'''
        model_file_path = '../file_data/aaai/new/slt_model'  # Model file path
        map_file_path = '../file_data/aaai/new/slt_encoder.tsv'
        if version == 'wiki':
            model_file_path = 'file_data/wiki-590k/slt_model'
            map_file_path = 'file_data/wiki-590k/slt_encoder.tsv'
        system = TangentCFTBackEnd(
            config_file=None, data_set=None, query_formulas=None)
        system.load_model(map_file_path=map_file_path,
                          model_file_path=model_file_path)
        return system.get_formula_vectors(query_formula)

    def get_vector_of_vocab(self, vocab):
        '''
        根据词汇从 embeddings 中读取向量
        :param vocab: 词汇在词汇表中的索引
        '''
        # 读取词表
        vocab_pickle = open(self.cache_file_pickle, 'rb')
        word2index, a = pickle.load(vocab_pickle)
        vocab_pickle.close()
        vocab2index = word2index[vocab]

        # 读取词向量
        embedding_pickle = open(self.cache_embedding_file, 'rb')
        data_embedding = pickle.load(embedding_pickle)
        embedding_pickle.close()

        return data_embedding[vocab2index]

    def get_vector_of_label(self, label_id, cache_embedding_file=None):
        '''
        根据标签从 label embeddings 中读取向量
        :param label: 标签id
        '''
        # 读取标签表
        vocab_pickle = open(self.cache_file_pickle, 'rb')
        a, label2index = pickle.load(vocab_pickle)
        vocab_pickle.close()
        idx = label2index[label_id]

        # 读取标签向量
        embedding_pickle = open(cache_embedding_file, 'rb')
        data_embedding = pickle.load(embedding_pickle)
        embedding_pickle.close()

        return data_embedding[idx]
