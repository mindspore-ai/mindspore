mindspore.Tensor.take
=====================

.. py:method:: mindspore.Tensor.take(indices, axis=None, mode='clip') -> Tensor

    在指定维度上获取Tensor中的元素。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：删除 `take(indices, axis=None, mode='clip')` 与 `take(index)` 重载，仅支持 `take(index)`。

    参数：
        - **indices** (Tensor) - 待提取的值的shape为 :math:`(Nj...)` 的索引。
        - **axis** (int, 可选) - 在指定维度上选择值。默认情况下，使用展开的输入数组。默认值： ``None`` 。
        - **mode** (str, 可选) - 支持 ``'raise'`` 、 ``'wrap'`` 、 ``'clip'`` 。

          - ``raise``：抛出错误。
          - ``wrap``：绕接。
          - ``clip``：裁剪到范围。 ``clip`` 模式意味着所有过大的索引都会被在指定轴方向上指向最后一个元素的索引替换。注：这将禁用具有负数的索引。

          默认值： ``'clip'`` 。

    返回：
        Tensor，索引的结果。

    异常：
        - **ValueError** - `axis` 超出范围，或 `mode` 被设置为 ``'raise'`` 、 ``'wrap'`` 和 ``'clip'`` 以外的值。

    .. py:method:: mindspore.Tensor.take(index) -> Tensor
        :noindex:

    选取给定索引 `index` 处的 `self` 元素。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **index** (LongTensor) - 输入张量的索引张量。

    返回：
        Tensor，shape与索引的shape相同。

    异常：
        - **TypeError** - 如果 `index` 的数据类型不是long。
