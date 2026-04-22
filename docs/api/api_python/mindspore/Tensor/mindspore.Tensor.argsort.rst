mindspore.Tensor.argsort
=========================

.. py:method:: mindspore.Tensor.argsort(axis=-1, descending=False) -> Tensor

    按指定顺序对 `self` 沿给定维度进行排序，并返回排序后的索引。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：删除 ``argsort(axis=-1, descending=False)`` 重载，仅支持 ``argsort(dim=-1, descending=False, stable=False)`` 。

    参数：
        - **axis** (int，可选) - 指定排序的轴。默认值： ``-1`` ，表示指定最后一维。当前Ascend后端只支持对最后一维进行排序。
        - **descending** (bool，可选) - 输出顺序。如果 `descending` 为 ``True`` ，按照元素值降序排序，否则升序排序。默认值： ``False`` 。

    返回：
        Tensor，排序后 `self` 的索引。数据类型为int32。

    .. py:method:: mindspore.Tensor.argsort(dim=-1, descending=False, stable=False) -> Tensor
        :noindex:

    按指定顺序对 `self` 沿给定维度进行排序，并返回排序后的索引。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **dim** (int，可选) - 指定排序的轴。默认值： ``-1`` ，表示指定最后一维。当前Ascend后端只支持对最后一维进行排序。
        - **descending** (bool，可选) - 输出顺序。如果 `descending` 为 ``True`` ，按照元素值降序排序，否则升序排序。默认值： ``False`` 。
        - **stable** (bool，可选) - 是否使用稳定排序算法。 默认值： ``False`` 。

    返回：
        Tensor，排序后 `self` 的索引。数据类型为int64。

    异常：
        - **ValueError** - 如果 `dim` 的设定值超出了范围。
        - **TypeError** - 如果 `dim` 的数据类型不是int32。
        - **TypeError** - 如果 `descending` 的数据类型不是bool。
        - **TypeError** - 如果 `stable` 的数据类型不是bool。
