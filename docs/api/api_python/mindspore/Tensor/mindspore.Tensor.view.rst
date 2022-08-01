mindspore.Tensor.view
=====================

.. py:method:: mindspore.Tensor.view(*shape)

    根据输入shape重新创建一个Tensor，与原Tensor数据相同。该方法与reshape方法相同，都是依靠底层reshape算子实现的。

    参数：
        - **shape** (Union[tuple(int), int]) - 输出Tensor的维度。

    返回：
        Tensor，具有与入参 `shape` 相同的维度。