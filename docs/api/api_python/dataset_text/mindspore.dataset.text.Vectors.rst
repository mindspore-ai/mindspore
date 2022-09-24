mindspore.dataset.text.Vectors
===============================

.. py:class:: mindspore.dataset.text.Vectors

    用于将tokens映射到矢量的Vectors对象。

    .. py:method:: from_file(file_path, max_vectors=None)

        从文件构建Vectors向量。

        参数：
            - **file_path** (str) - 包含向量的文件的路径。
            - **max_vectors** (int，可选) - 用于限制加载的预训练向量的数量。
              大多数预训练的向量集是按词频降序排序的。因此，在如果内存不能存放整个向量集，或者由于其他原因不需要，
              可以传递 `max_vectors` 限制加载数量。默认值：None，无限制。

        返回：
            Vectors， 根据文件构建的Vectors向量。
