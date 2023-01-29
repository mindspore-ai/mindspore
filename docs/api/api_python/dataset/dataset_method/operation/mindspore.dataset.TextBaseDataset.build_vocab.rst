mindspore.dataset.TextBaseDataset.build_vocab
=============================================

.. py:method:: mindspore.dataset.TextBaseDataset.build_vocab(columns, freq_range, top_k, special_tokens, special_first)

    迭代源数据集对象获取数据并构建词汇表。
    源数据集要求的是文本类数据集。

    收集数据集中所有的不重复单词。返回 `top_k` 个最常见的单词组成的词汇表（如果指定了 `top_k` ）。

    .. note:: mindspore.dataset.Dataset.build_vocab 从2.0版本开始弃用。请使用mindspore.dataset.text.Vocab.from_dataset代替。

    参数：
        - **columns** (Union[str, list[str]]) - 指定 `build_vocab` 操作的输入列，会从该列获取数据构造词汇表。
        - **freq_range** (tuple[int]) - 由(min_frequency, max_frequency)组成的整数元组，代表词汇出现的频率范围，在这个频率范围的词汇会被保存下来。
          取值范围需满足：0 <= min_frequency <= max_frequency <= 单词总数，其中min_frequency、max_frequency的默认值分别设置为0、单词总数。
        - **top_k** (int) - 使用 `top_k` 个最常见的单词构建词汇表。假如指定了参数 `freq_range` ，则优先统计给定频率范围内的词汇，再根据参数 `top_k` 选取最常见的单词构建词汇表。
          如果 `top_k` 的值大于单词总数，则取所有单词构建词汇表。
        - **special_tokens** (list[str]) - 指定词汇表的特殊标记（special token），如 '[UNK]'、 '[SEP]'。
        - **special_first** (bool) - 是否将参数 `special_tokens` 指定的特殊标记添加到词汇表的开头。如果为True则放到开头，否则放到词汇表的结尾。

    返回：
        构建好的词汇表。
