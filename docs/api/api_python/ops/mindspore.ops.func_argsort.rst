mindspore.ops.argsort
======================

.. py:function:: mindspore.ops.argsort(input, axis=-1, descending=False)

    返回按指定轴对tensor进行排序后的索引。

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `axis` 将重命名为 `dim`，并新增 `stable` 参数，
        接口签名将变更为 ``mindspore.ops.argsort(input, dim=-1, descending=False, stable=False)`` 。

    .. note::
        当前Ascend后端只支持对最后一维进行排序。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int) - 指定轴。默认 ``-1`` 。
        - **descending** (bool) - 指定排序（升序或降序）。默认 ``False`` 。

    返回：
        Tensor
