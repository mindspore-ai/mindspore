.. py:method:: build_sentencepiece_vocab(columns, vocab_size, character_coverage, model_type, params)

    迭代源数据集对象获取数据并构建SentencePiece词汇表。
    源数据集要求的是文本类数据集。

    参数：
        - **columns** (list[str]) - 指定 `build_sentencepiece_vocab` 操作的输入列，会从该列获取数据构造词汇表。
        - **vocab_size** (int) - 词汇表的容量。
        - **character_coverage** (float) - 模型涵盖的字符百分比，必须介于0.98和1.0之间。
          对于具有丰富字符集的语言，如日语或中文字符集，推荐使用0.9995；对于其他字符集较小的语言，比如英语或拉丁文，推荐使用1.0。
        - **model_type** (SentencePieceModel) - 训练的SentencePiece模型类型，可取值为'SentencePieceModel.UNIGRAM'、'SentencePieceModel.BPE'、'SentencePieceModel.CHAR'或'SentencePieceModel.WORD'。
          当取值为'SentencePieceModel.WORD'时，输入的数据必须进行预分词（pretokenize）。默认值：SentencePieceModel.UNIGRAM。
        - **params** (dict) - 如果希望使用SentencePiece的其他参数，可以构造一个dict进行传入，键为SentencePiece库接口的输入参数名，值为参数值。

    返回：
        构建好的SentencePiece词汇表。

.. py:method:: build_vocab(columns, freq_range, top_k, special_tokens, special_first)

    迭代源数据集对象获取数据并构建词汇表。
    源数据集要求的是文本类数据集。

    收集数据集中所有的不重复单词。返回 `top_k` 个最常见的单词组成的词汇表（如果指定了 `top_k` ）。

    参数：
        - **columns** (Union[str, list[str]]) - 指定 `build_vocab` 操作的输入列，会从该列获取数据构造词汇表。
        - **freq_range** (tuple[int]) - 由(min_frequency, max_frequency)组成的整数元组，代表词汇出现的频率范围，在这个频率范围的词汇会被保存下来。
          取值范围需满足：0 <= min_frequency <= max_frequency <= 单词总数，其中min_frequency、max_frequency的默认值分别设置为0、单词总数。
        - **top_k** (int) - 使用 `top_k` 个最常见的单词构建词汇表。假如指定了参数 `freq_range` ，则优先统计给定频率范围内的词汇，再根据参数 `top_k` 选取最常见的单词构建词汇表。
          如果 `top_k` 的值大于单词总数，则取所有单词构建词汇表。
        - **special_tokens** (list[str]) - 指定词汇表的特殊标记（special token），如'[UNK]'、'[SEP]'。
        - **special_first** (bool) - 是否将参数 `special_tokens` 指定的特殊标记添加到词汇表的开头。如果为True则放到开头，否则放到词汇表的结尾。

    返回：
        构建好的词汇表。
