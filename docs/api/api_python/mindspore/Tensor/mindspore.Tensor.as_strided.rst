mindspore.Tensor.as_strided
============================

.. py:method:: mindspore.Tensor.as_strided(self, shape=None, strides=None, subok=False)

    创建现有张量的视图，具有指定的`shape`、`stead`和`subok`。

    参数：
        - **shape** (tuple或ints) - 输出张量的形状
        - **strides** (tuple或ints) - 输出张量的步幅
        - **subok** (int，可选) - 输出张量的偏移量

    返回：
        Tensor的视图。

    平台：
        ``Ascemd`` ``GPU`` `` CPU``
