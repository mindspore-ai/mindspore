mindspore.ops.truncate_mod
==========================

.. py:function:: mindspore.ops.truncate_mod(x, y)

    将 `x` 和 `y` 逐元素取模。

    支持隐式类型转换，支持广播。

    .. warning::
        - 从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：接口名称将改为 `mindspore.ops.fmod`，
          参数 `x` 和 `y` 将分别重命名为 `input` 和 `other` 。
        - 输入数值不能为0。
        - 当输入含有超过2048个元素时，该操作不能保证千分之二的精度要求。
        - 由于架构不同，该算子在NPU和CPU上的计算结果可能不一致。
        - 若shape为（D1、D2...、Dn），则D1*D2...*DN<=1000000，n<=8。

    参数：
        - **x** (Union[Tensor, Number, bool]) - 第一个输入tensor。
        - **y** (Union[Tensor, Number, bool]) - 第二个输入tensor。

    返回：
        Tensor
