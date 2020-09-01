import numpy as np 


class RecMetrics(object):
    @staticmethod
    def precision(predict,truth):
        if len(predict) == 0:
            return 0.0
        else:
            true_positive = set(predict).intersection(truth)
            return len(true_positive)/len(set(truth))

    @staticmethod
    def recall(predict,truth):
        if len(truth) == 0:
            return 0.0
        else:
            true_positive = set(predict).intersection(truth)
            return len(true_positive)/len(set(truth))

    @classmethod
    def _p_k(cls,predict,truth,k):
        predict_ = predict[:k]
        return cls.precision(predict_,truth)
    @classmethod
    def _r_k(cls,predict,truth,k):
        predict_ = predict[:k]
        return cls.recall(predict_,truth)
    @staticmethod
    def coverage(predict_whole,truth_whole):
        coverage = set(predict_whole).intersection(set(truth_whole))
        return len(coverage)/len(set(truth_whole))
    @staticmethod
    def personalization(matrix):
        matrix_sparse = sparse.csr_matrix(matrix)
        similarity = cosine_similarity(matrix_sparse)
        upper_triangle_index = np.triu_indices(similarity.shape[0],1)
        upper_triangle = similarity[upper_triangle_index]
        personalization = 1-np.mean(upper_triangle)
        return personalization
    @staticmethod
    def novelty():
        pass
class RecMetricsMap(RecMetrics):
    @classmethod
    def _p_k_map(cls,predict_col,truth_col,k):
        def _p_k_tmp(row):
            predict = row[predict_col]
            truth = row[truth_col]
            return cls._p_k(predict,truth,k)
        return _p_k_tmp
    @classmethod
    def _r_k_map(cls,predict_col,truth_col,k):
        def _r_k_tmp(row):
            predict = row[predict_col]
            truth = row[truth_col]
            return cls._r_k(predict,truth,k)
        return _r_k_tmp

