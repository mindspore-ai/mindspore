mindspore.ops.take_along_dim
===============================

.. py:function:: mindspore.ops.take_along_dim(input, indices, dim=None)

    在指定维度上根据索引获取Tensor中的元素。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **indices** (Tensor) - 输入的索引，必须为可以扩展为与`input`shape一致的整数型Tensor。
        - **dim** (int, 可选) - 选择的维度。默认值： ``None`` ，取值为 ``None`` 时，`input` 会展开成一维的Tensor后进行取值。

    返回：
        Tensor，索引的结果。

    异常：
        - **ValueError** - ``indices`` 不可扩展为与`input`shape相同的Tensor。
