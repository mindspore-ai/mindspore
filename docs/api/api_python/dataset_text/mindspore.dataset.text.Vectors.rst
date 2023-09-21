mindspore.dataset.text.Vectors
===============================

.. py:class:: mindspore.dataset.text.Vectors

    预训练词向量。

    .. py:method:: from_file(file_path, max_vectors=None)

        加载预训练向量集文件。

        参数：
            - **file_path** (str) - 预训练向量集文件路径。
            - **max_vectors** (int，可选) - 加载预训练向量的数量上限。
              大多数预训练向量集是按词频降序排列的。因此，如果内存不足以存放整个向量集，或者出于其他原因，
              可以通过该值限制加载的向量数量。默认值： ``None`` ，没有上限。

        返回：
            CharNGram，CharNGram 预训练词向量。

        异常：
            - **TypeError** - 当 `file_path` 不为str类型。
            - **RuntimeError** - 当 `file_path` 文件路径不存在或没有访问权限。
            - **TypeError** - 当 `max_vectors` 不为int类型。
            - **ValueError** - 当 `max_vectors` 为负数。
