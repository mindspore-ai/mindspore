.. py:method:: build_sentencepiece_vocab(columns, vocab_size, character_coverage, model_type, params)

    用于从源数据集对象创建句子词表的函数。

    **参数：**

    - **columns** (list[str])：指定从哪一列中获取单词。
    - **vocab_size** (int)：词汇表大小。
    - **character_coverage** (int)：模型涵盖的字符百分比，必须介于0.98和1.0之间。默认值如0.9995，适用于具有丰富字符集的语言，如日语或中文字符集；1.0适用于其他字符集较小的语言，比如英语或拉丁文。
    - **model_type** (SentencePieceModel)：模型类型，枚举值包括unigram（默认值）、bpe、char及word。当类型为word时，输入句子必须预先标记。
    - **params** (dict)：依据原始数据内容构建祠表的附加参数，无附加参数时取值可以是空字典。

    **返回：**

    SentencePieceVocab，从数据集构建的词汇表。

    **样例：**

    >>> from mindspore.dataset.text import SentencePieceModel
    >>>
    >>> # DE_C_INTER_SENTENCEPIECE_MODE 是一个映射字典
    >>> from mindspore.dataset.text.utils import DE_C_INTER_SENTENCEPIECE_MODE
    >>> dataset = ds.TextFileDataset("/path/to/sentence/piece/vocab/file", shuffle=False)
    >>> dataset = dataset.build_sentencepiece_vocab(["text"], 5000, 0.9995,
    ...                                             DE_C_INTER_SENTENCEPIECE_MODE[SentencePieceModel.UNIGRAM],
    ...                                             {})