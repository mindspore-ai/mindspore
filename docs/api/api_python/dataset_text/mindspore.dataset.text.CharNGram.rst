mindspore.dataset.text.CharNGram
=================================

.. py:class:: mindspore.dataset.text.CharNGram

    CharNGram对象，用于将 `tokens` 映射到预训练的向量中。

    .. py:method:: from_file(file_path, max_vectors=None)

        从文件构建 `CharNGram` 向量。

        参数：
            - **file_path** (str) - 包含 `CharNGram` 向量的文件路径。
            - **max_vectors** (int，可选) - 用于限制加载的预训练向量的数量。
              大多数预训练的向量集是按词频降序排序的。因此，如果内存不能存放整个向量集，或者由于其他原因不需要，
              可以传递 `max_vectors` 限制加载数量。默认值：None，无限制。

        返回：
            CharNGram， 根据文件构建的CharNGram向量。

        异常：
            - **RuntimeError** - `file_path` 参数所指向的文件非法或者包含的数据异常。
            - **ValueError** - `max_vectors` 参数值错误。
            - **TypeError** - `max_vectors` 参数不是整数类型。
