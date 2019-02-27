import string
from collections import Counter


def bow_treat(documents):
    """
    英文 BoW
    :param documents: 字符串列表
    :return:
    """
    # 单词转换为小写
    lower_case_documents = [x.lower() for x in documents]
    # 删除标点符号
    sans_punctuation_documents = [x.translate(str.maketrans('', '', string.punctuation)) for x in lower_case_documents]
    # 单词化
    preprocessed_documents = [x.split(' ') for x in sans_punctuation_documents]
    # 计算频率
    frequency_list = [Counter(x) for x in preprocessed_documents]

    return frequency_list
