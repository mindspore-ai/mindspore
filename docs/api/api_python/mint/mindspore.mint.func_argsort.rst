mindspore.mint.argsort
======================

.. py:function:: mindspore.mint.argsort(input, dim=-1, descending=False, stable=False)

    返回按指定维度对tensor进行排序后的索引。

    .. warning::
        这是一个实验性API，后续可能修改或删除

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (int，可选) - 指定维度。默认 ``-1`` 。
        - **descending** (bool，可选) - 指定排序（升序或降序）。默认 ``False``。
        - **stable** (bool，可选) - 控制等效元素的相对顺序。默认 ``False``。

    返回：
        Tensor