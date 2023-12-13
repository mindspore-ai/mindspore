mindspore.dataset.text.Vocab
=============================

.. py:class:: mindspore.dataset.text.Vocab

    创建用于训练NLP模型的Vocab。

    Vocab是数据集中可能出现的所有Token的集合，保存了各Token与其ID之间的映射关系。

    .. py:method:: from_dataset(dataset, columns=None, freq_range=None, top_k=None, special_tokens=None, special_first=True)
        :classmethod:

        从给定数据集创建Vocab。

        以数据集中的样本作为语料库创建Vocab，其中的Token按词频从高到低进行排列，具有相同词频的Token则按字母序进行排列。

        参数：
            - **dataset** (Dataset) - 用于创建Vocab的数据集。
            - **columns** (list[str]，可选) - 用于创建Vocab的数据列列名。默认值： ``None`` ，使用所有列。
            - **freq_range** (tuple[int, int]，可选) - 用于创建Vocab的词频范围，需包含两个元素，分别表示最低频率与最高频率，在此范围内的Token将会保留。
              当最低或最高频率为None时，表示没有最低或最高频率限制。默认值： ``None`` ，没有词频范围限制。
            - **top_k** (int，可选) - 只选取词频最高的前指定个Token构建Vocab。此操作将在词频筛选后进行。如果该值大于总词数，则所有Token都将保留。默认值： ``None`` ，没有Token个数限制。
            - **special_tokens** (list[str]，可选) - 追加到Vocab中的Token列表。默认值： ``None`` ，不追加Token。
            - **special_first** (bool，可选) - 是否将追加的Token添加到Vocab首部，否则添加到Vocab尾部。默认值： ``True`` 。

        返回：
            Vocab，从数据集创建的Vocab。

        异常：
            - **TypeError** - 当 `columns` 不为list[str]类型。
            - **TypeError** - 当 `freq_range` 不为tuple[int, int]类型。
            - **ValueError** - 当 `freq_range` 中的元素值为负数。
            - **TypeError** - 当 `top_k` 不为int类型。
            - **ValueError** - 当 `top_k` 不为正数。
            - **TypeError** - 当 `special_tokens` 不为list[str]类型。
            - **ValueError** - 当 `special_tokens` 中存在重复元素。
            - **TypeError** - 当 `special_first` 不为bool类型。

    .. py:method:: from_dict(word_dict)
        :classmethod:

        从给定字典创建Vocab。

        参数：
            - **word_dict** (dict[str, int]) - 存储各Token与其ID之间映射关系的字典。

        返回：
            Vocab，从字典创建的Vocab。

        异常：
            - **TypeError** - 当 `word_dict` 不为dict[str, int]类型。
            - **ValueError** - 当 `word_dict` 中的键值为负数。

    .. py:method:: from_file(file_path, delimiter="", vocab_size=None, special_tokens=None, special_first=True)
        :classmethod:

        从给定文件创建Vocab。

        参数：
            - **file_path** (str) - 用于创建Vocab的文件路径。
            - **delimiter** (str，可选) - 文件行中Token的分隔符。分隔符前的字符串将被视为一个Token。默认值： ``""`` ，整行将被视为一个Token。
            - **vocab_size** (int，可选) - Vocab包含的Token数上限。默认值： ``None`` ，没有Token数上限。
            - **special_tokens** (list[str]，可选) - 追加到Vocab中的Token列表。默认值： ``None`` ，不追加Token。
            - **special_first** (bool，可选) - 是否将追加的Token添加到Vocab首部，否则添加到Vocab尾部。默认值： ``True`` 。

        返回：
            Vocab，从文件创建的Vocab。

        异常：
            - **TypeError** - 当 `file_path` 不为str类型。
            - **TypeError** - 当 `delimiter` 不为str类型。
            - **ValueError** - 当 `vocab_size` 不为正数。
            - **TypeError** - 当 `special_tokens` 不为list[str]类型。
            - **ValueError** - 当 `special_tokens` 中存在重复元素。
            - **TypeError** - 当 `special_first` 不为bool类型。


    .. py:method:: from_list(word_list, special_tokens=None, special_first=True)
        :classmethod:

        从给定Token列表创建Vocab。

        参数：
            - **word_list** (list[str]) - 用于创建Vocab的Token列表。
            - **special_tokens** (list[str]，可选) - 追加到Vocab中的Token列表。默认值： ``None`` ，不追加Token。
            - **special_first** (bool，可选) - 是否将追加的Token添加到Vocab首部，否则添加到Vocab尾部。默认值： ``True`` 。

        返回：
            Vocab，从Token列表创建的Vocab。

        异常：
            - **TypeError** - 当 `word_list` 不为list[str]类型。
            - **ValueError** - 当 `word_list` 中存在重复元素。
            - **TypeError** - 当 `special_tokens` 不为list[str]类型。
            - **ValueError** - 当 `special_tokens` 中存在重复元素。
            - **TypeError** - 当 `special_first` 不为bool类型。

    .. py:method:: ids_to_tokens(ids)

        查找指定ID对应的Token。

        参数：
            - **ids** (Union[int, list[int], numpy.ndarray]) - 待查找的某个ID或ID列表。若ID不存在，则返回空字符串。

        返回：
            Union[str, list[str]]，指定ID对应的Token。

        异常：
            - **TypeError** - 当 `ids` 不为Union[int, list[int], numpy.ndarray]类型。
            - **ValueError** - 当 `ids` 中的元素为负数。

    .. py:method:: tokens_to_ids(tokens)

        查找指定Token对应的ID。

        参数：
            - **tokens** (Union[str, list[str], numpy.ndarray]) - 待查找的某个Token或Token列表。如果Token不存在，则返回-1。

        返回：
            Union[int, list[int]]，指定Token对应的ID。

        异常：
            - **TypeError** - 当 `tokens` 不为Union[str, list[str], numpy.ndarray]类型。

    .. py:method:: vocab()

        获取Token与其ID之间映射关系的字典。

        返回：
            dict[str, int]，Token与ID之间映射关系的字典。
