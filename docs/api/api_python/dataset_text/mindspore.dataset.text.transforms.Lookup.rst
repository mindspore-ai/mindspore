mindspore.dataset.text.transforms.Lookup
========================================

.. py:class:: mindspore.dataset.text.transforms.Lookup(vocab, unknown_token=None, data_type=mstype.int32)

    根据词表，将分词标记(token)映射到其索引值(id)。

    **参数：**

    - **vocab** (Vocab) - 词表对象，用于存储分词和索引的映射。
    - **unknown_token** (str, 可选) - 备用词汇，用于在单词不在词汇表中的情况。
      即如果单词不在词汇表中，则查找结果将替换为 `unknown_token` 的值。
      如果如果单词不在词汇表中，且未指定 `unknown_token` ，将抛出运行时错误。默认值：None，不指定该参数。
    - **data_type** (mindspore.dtype, 可选): Lookup输出的数据类型，默认值：mindspore.int32。

    **异常：**
      
    - **TypeError** - 参数 `vocab` 类型不为 text.Vocab。
    - **TypeError** - 参数 `unknown_token` 类型不为string。
    - **TypeError** - 参数 `data_type` 类型不为 mindspore.dtype。
