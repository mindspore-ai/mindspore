mindspore.Tensor.flatten
========================

.. py:method:: mindspore.Tensor.flatten(order='C')

    返回展开成一维的Tensor的副本。

    参数：
        - **order** (str, 可选) - 可以在'C'和'F'之间进行选择。'C'表示按行优先（C风格）顺序展开。'F'表示按列优先顺序（Fortran风格）进行扁平化。仅支持'C'和'F'。默认值：'C'。

    返回：
        Tensor，具有与输入相同的数据类型。

    异常：
        - **TypeError** - `order` 不是字符串类型。
        - **ValueError** - `order` 是字符串类型，但不是'C'或'F'。

    比如：
        :func:`mindspore.Tensor.reshape`：在不改变数据的情况，改变Tensor的shape。

        :func:`mindspore.Tensor.ravel`：返回一个连续扁平化的Tensor。