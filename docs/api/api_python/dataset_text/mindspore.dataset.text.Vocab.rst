mindspore.dataset.text.Vocab
=============================

.. py:class:: mindspore.dataset.text.Vocab

    用于查找单词的Vocab对象。

    它包含一个映射，将每个单词（str）映射到一个ID（int）。

    .. py:method:: from_dataset(dataset, columns=None, freq_range=None, top_k=None, special_tokens=None, special_first=True)

        通过数据集构建Vocab对象。

        这将收集数据集中的所有唯一单词，并在freq_range中用户指定的频率范围内返回一个vocab。如果没有单词在该频率上，用户将收到预警信息。
        vocab中的单词按最高频率到最低频率的顺序进行排列。具有相同频率的单词将按词典顺序进行排列。

        **参数：**

        - **dataset** (Dataset) - 表示要从中构建vocab的数据集。
        - **columns** (list[str]，可选) - 表示要从中获取单词的列名。它可以是列名的列表，默认值：None。如果没有列是string类型，将返回错误。
        - **freq_range** (tuple，可选) - 表示整数元组（min_frequency，max_frequency）。频率范围内的单词将被保留。0 <= min_frequency <= max_frequency <= total_words。min_frequency=0等同于min_frequency=1。max_frequency > total_words等同于max_frequency = total_words。min_frequency和max_frequency可以为None，分别对应于0和total_words，默认值：None。
        - **top_k** (int，可选) - `top_k` 大于0。要在vocab中 `top_k` 建立的单词数量表示取用最频繁的单词。 `top_k` 在 `freq_range` 之后取用。如果没有足够的 `top_k` ，所有单词都将被取用,默认值：None。
        - **special_tokens** (list，可选) - 表示字符串列表。每个字符串都是一个特殊的标记。例如，special_tokens=["<pad>","<unk>"]，默认值：None，表示不添加特殊标记。
        - **special_first** (bool，可选) - 表示是否添加 `special_tokens` 到vocab。如果指定了 `special_tokens` 并将 `special_first` 设置为True，则添加special_tokens，默认值：True。

        **返回：**

        Vocab，从数据集构建的Vocab对象。

    .. py:method:: from_dict(word_dict)

        通过字典构建Vocab对象。

        **参数：**

        - **word_dict** (dict) - 字典包含word和ID对，其中 `word` 应是string类型， `ID` 应是int类型。至于 `ID` ，建议从0开始并且不断续。如果 `ID` 为负数，将引发ValueError。

        **返回：**

        Vocab，从字典构建的Vocab对象。

    .. py:method:: from_file(file_path, delimiter='', vocab_size=None, special_tokens=None, special_first=True)

        通过文件构建Vocab对象。

        **参数：**

        - **file_path** (str) - 表示包含vocab列表的文件的路径。
        - **delimiter** (str，可选) - 表示用来分隔文件中每一行的分隔符。第一个元素被视为单词，默认值：""。
        - **vocab_size** (int，可选) - 表示要从 `file_path` 读取的字数，默认值：None，表示读取所有的字。
        - **special_tokens** (list，可选) - 表示字符串的列表。每个字符串都是一个特殊标记，例如special_tokens=["<pad>","<unk>"]，默认值：None，表示不添加特殊标记）。
        - **special_first** (list，可选) - 表示是否添加 `special_tokens` 到vocab。如果指定了 `special_tokens` 并将 `special_first` 设置为True，则添加 `special_tokens` ，默认值：True。

        **返回：**

        Vocab，从文件构建的Vocab对象。


.. py:method:: from_list(word_list, special_tokens=None, special_first=True)

        通过单词列表构建Vocab对象。

        **参数：**

        - **word_list** (list) - 表示字符串列表，其中每个元素都是type类型的单词。
        - **special_tokens** (list，可选) - 表示字符串的列表。每个字符串都是一个特殊标记，例如special_tokens=["<pad>","<unk>"]，默认值：None，表示不添加特殊标记。
        - **Special_first** (bool，可选) - 表示是否添加 `special_tokens` 到vocab。如果指定了 `special_tokens` 并将 `special_first` 设置为True，则添加 `special_tokens` ，默认值：True。

        **返回：**

        Vocab，从单词列表构建的Vocab对象。
