mindspore.Tensor.take
=====================

.. py:method:: mindspore.Tensor.take(indices, axis=None, mode='clip')

    在指定维度上获取Tensor中的元素。

    参数：
        - **indices** (Tensor) - 待提取的值的shape为 `(Nj...)` 的索引。
        - **axis** (int, 可选) - 在指定维度上选择值。默认情况下，使用展开的输入数组。默认值：None。
        - **mode** ('raise', 'wrap', 'clip', 可选)

          - raise：抛出错误。
          - wrap：绕接。
          - clip：裁剪到范围。 `clip` 模式意味着所有过大的索引都会被在指定轴方向上指向最后一个元素的索引替换。注：这将禁用具有负数的索引。默认值：`clip` 。

    返回：
        Tensor，索引的结果。

    异常：
        - **ValueError** - `axis` 超出范围，或 `mode` 被设置为'raise'、'wrap'和'clip'以外的值。