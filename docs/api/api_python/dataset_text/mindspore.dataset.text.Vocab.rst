mindspore.dataset.text.Vocab
=============================

.. py:class:: mindspore.dataset.text.Vocab

    用于查找单词的Vocab对象。

    它包含一个映射，将每个单词（str）映射到一个ID（int）。

    .. py:method:: from_dataset(dataset, columns=None, freq_range=None, top_k=None, special_tokens=None, special_first=True)

        通过数据集构建Vocab对象。

        获得数据集中的所有唯一单词，并在 `freq_range` 中用户指定的频率范围内返回一个vocab。如果没有单词在该频率上，用户将收到预警信息。
        vocab中的单词按最高频率到最低频率的顺序进行排列。具有相同频率的单词将按词典顺序进行排列。

        参数：
            - **dataset** (Dataset) - 表示要从中构建vocab的数据集。
            - **columns** (list[str]，可选) - 表示要从中获取单词的列名。它可以是列名的列表。默认值：None。
            - **freq_range** (tuple，可选) - 表示整数元组（min_frequency，max_frequency）。频率范围内的单词将被保留。0 <= min_frequency <= max_frequency <= total_words。min_frequency=0等同于min_frequency=1。max_frequency > total_words等同于max_frequency = total_words。min_frequency和max_frequency可以为None，分别对应于0和total_words。默认值：None。
            - **top_k** (int，可选) - `top_k` 大于0。要在vocab中 `top_k` 建立的单词数量表示取用最频繁的单词。 `top_k` 在 `freq_range` 之后取用。如果没有足够的 `top_k` ，所有单词都将被取用。默认值：None。
            - **special_tokens** (list，可选) - 特殊分词列表，如常用的"<pad>"、"<unk>"等。默认值：None，表示不添加特殊分词（token）。
            - **special_first** (bool，可选) - 表示是否将 `special_tokens` 中的特殊分词添加到词典的最前面。如果为True则将 `special_tokens` 添加到词典的最前，否则添加到词典的最后。默认值：True。

        返回：
            Vocab，从数据集构建的Vocab对象。

    .. py:method:: from_dict(word_dict)

        通过字典构建Vocab对象。

        参数：
            - **word_dict** (dict) - 字典包含word和ID对，其中 `word` 应是string类型， `ID` 应是int类型。至于 `ID` ，建议从0开始并且不断续。如果 `ID` 为负数，将引发ValueError。

        返回：
            Vocab，从字典构建的Vocab对象。

    .. py:method:: from_file(file_path, delimiter="", vocab_size=None, special_tokens=None, special_first=True)

        通过文件构建Vocab对象。

        参数：
            - **file_path** (str) - 表示包含vocab文件路径的一个列表。
            - **delimiter** (str，可选) - 表示用来分隔文件中每一行的分隔符。第一个元素被视为单词。默认值：""。
            - **vocab_size** (int，可选) - 表示要从 `file_path` 读取的字数。默认值：None，表示读取所有的字。
            - **special_tokens** (list，可选) - 特殊分词列表，如常用的"<pad>"、"<unk>"等。默认值：None，表示不添加特殊分词（token）。
            - **special_first** (bool，可选) - 表示是否将 `special_tokens` 中的特殊分词添加到词典的最前面。如果为True则将 `special_tokens` 添加到词典的最前，否则添加到词典的最后。默认值：True。

        返回：
            Vocab，从文件构建的Vocab对象。

    .. py:method:: from_list(word_list, special_tokens=None, special_first=True)

        从单词列表构建一个vocab对象。

        参数：
            - **word_list** (list) - 输入单词列表，每个单词需要为字符串类型。
            - **special_tokens** (list，可选) - 特殊分词列表，如常用的"<pad>"、"<unk>"等。默认值：None，表示不添加特殊分词（token）。
            - **special_first** (bool，可选) - 表示是否将 `special_tokens` 中的特殊分词添加到词典的最前面。如果为True则将 `special_tokens` 添加到词典的最前，否则添加到词典的最后。默认值：True。

        返回：
            Vocab，从单词列表构建的Vocab对象。

    .. py:method:: ids_to_tokens(ids)

        将输入索引转换为对应的分词，支持传入单个索引或一个包含多个索引的序列。如果索引不存在，则返回空字符串。

        参数：
            - **ids** (Union[int, list[int]]) - 要转换为分词的分词索引（或分词的索引序列）。

        返回：
            解码的分词（token）。

    .. py:method:: tokens_to_ids(tokens)

        将输入分词(token)转换为对应的索引(id)，支持传入单个分词或一个包含多个分词的列表。如果分词不存在，则返回-1。

        参数：
            - **tokens** (Union[str, list[str]]) - 一个或多个要转换为分词（token）id(s)的分词（token）。

        返回：
            分词（token）id或分词（token）id列表。

    .. py:method:: vocab()

        获取dict类型的词汇表。

        返回：
            由word和id对组成的词汇表。
