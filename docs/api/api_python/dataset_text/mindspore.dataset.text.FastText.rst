mindspore.dataset.text.FastText
================================

.. py:class:: mindspore.dataset.text.FastText

    FastText 预训练词向量。

    通过 FastText ，可以创建一种无监督或有监督学习算法，以获得单词的向量表示。

    .. py:method:: from_file(file_path, max_vectors=None)

        加载 FastText 预训练向量集文件。

        参数：
            - **file_path** (str) - FastText 预训练向量集文件路径。文件后缀需为 `*.vec` 。
            - **max_vectors** (int，可选) - 加载预训练向量的数量上限。
              大多数预训练向量集是按词频降序排列的。因此，如果内存不足以存放整个向量集，或者出于其他原因，
              可以通过该值限制加载的向量数量。默认值： ``None`` ，没有上限。

        返回：
            FastText，FastText 预训练词向量。

        异常：
            - **TypeError** - 当 `file_path` 不为str类型。
            - **RuntimeError** - 当 `file_path` 文件路径不存在或没有访问权限。
            - **TypeError** - 当 `max_vectors` 不为int类型。
            - **ValueError** - 当 `max_vectors` 为负数。
