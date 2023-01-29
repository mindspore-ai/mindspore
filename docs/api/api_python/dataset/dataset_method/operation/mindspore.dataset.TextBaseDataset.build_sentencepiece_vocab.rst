mindspore.dataset.TextBaseDataset.build_sentencepiece_vocab
===========================================================

.. py:method:: mindspore.dataset.TextBaseDataset.build_sentencepiece_vocab(columns, vocab_size, character_coverage, model_type, params)

    迭代源数据集对象获取数据并构建SentencePiece词汇表。
    源数据集要求的是文本类数据集。

    .. note:: mindspore.dataset.Dataset.build_sentencepiece_vocab 从2.0版本开始弃用。请使用mindspore.dataset.text.SentencePieceVocab.from_dataset代替。

    参数：
        - **columns** (list[str]) - 指定 `build_sentencepiece_vocab` 操作的输入列，会从该列获取数据构造词汇表。
        - **vocab_size** (int) - 词汇表的容量。
        - **character_coverage** (float) - 模型涵盖的字符百分比，必须介于0.98和1.0之间。
          对于具有丰富字符集的语言，如日语或中文字符集，推荐使用0.9995；对于其他字符集较小的语言，比如英语或拉丁文，推荐使用1.0。
        - **model_type** (SentencePieceModel) - 训练的SentencePiece模型类型，可取值为 'SentencePieceModel.UNIGRAM'、 'SentencePieceModel.BPE'、 'SentencePieceModel.CHAR'或 'SentencePieceModel.WORD'。
          当取值为 'SentencePieceModel.WORD'时，输入的数据必须进行预分词（pretokenize）。默认值：SentencePieceModel.UNIGRAM。
        - **params** (dict) - 如果希望使用SentencePiece的其他参数，可以构造一个dict进行传入，键为SentencePiece库接口的输入参数名，值为参数值。

    返回：
        构建好的SentencePiece词汇表。
