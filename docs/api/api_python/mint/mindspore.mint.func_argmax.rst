mindspore.mint.argmax
=====================

.. py:function:: mindspore.mint.argmax(input) -> Tensor

    返回tensor的最大值索引。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor。

    .. py:function:: mindspore.mint.argmax(input, dim, keepdim=False) -> Tensor
        :noindex:

    返回tensor在指定维度上的最大值索引。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (int) - 指定维度。
        - **keepdim** (bool，可选) - 是否保留输出tensor的维度。默认 ``False`` 。

    返回：
        Tensor

    异常：
        - **ValueError** - 如果 `dim` 的设定值超出了范围。
