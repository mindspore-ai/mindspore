mindspore.ops.floor_mod
========================

.. py:function:: mindspore.ops.floor_mod(x, y)

    逐元素计算第一个输入除以第二个输入，并向下取余。

    .. warning::
        - 该接口已废弃，2.9.0（不含）之后版本将删除。
        - 输入 `y` 的元素不能等于0，否则将返回当前数据类型的最大值。
        - 当输入元素数量超过2048时，算子的精度不能保证千分之二的要求。
        - 由于架构不同，该算子在NPU和CPU上的计算结果可能不一致。
        - 如果shape表示为 :math:`(D1, D2 ..., Dn)` ，那么 D1\*D2... \*DN<=1000000,n<=8。

    如果两个输入数据类型不同，遵循隐式类型转换规则。
    输入必须是两个tensor或一个tensor和一个scalar。
    当输入是两个tensor时，其shape须可以进行广播，并且数据类型不能同时为bool。

    .. math::
        out_{i} =\text{floor}(x_{i} // y_{i})

    参数：
        - **x** (Union[Tensor, Number, bool]) - 第一个输入tensor。
        - **y** (Union[Tensor, Number, bool]) - 第二个输入tensor。

    返回：
        Tensor
