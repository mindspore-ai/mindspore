mindspore.Tensor.nansum
=======================

.. py:method:: mindspore.Tensor.nansum(dim=None, keepdim=False, *, dtype=None) -> Tensor

    计算输入Tensor指定维度元素的和，将非数字(NaNs)处理为零。

    .. warning::
        - 仅支持 Atlas A2 训练系列产品。
        - 这是一个实验性API，后续可能修改或删除。

    参数：
        - **dim** (Union[int, tuple(int)], 可选) - 求和的维度。取值范围[-rank(self), rank(self))。默认值： ``None`` ，对Tensor中的所有元素求和。
        - **keepdim** (bool, 可选) - 输出Tensor是否保持维度。默认值： ``False`` ，不保留维度。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 输出Tensor的类型。默认值： ``None`` 。

    返回：
        Tensor，输入Tensor指定维度的元素和，将非数字(NaNs)处理为零。

        - 如果 `dim` 为 ``None``，且 `keepdim` 为 ``False``，
          则输出一个零维Tensor，表示输入Tensor中所有元素的和。
        - 如果 `dim` 为int，值为2，并且 `keepdim` 为 ``False``，
          则输出的shape为： :math:`(self_1, self_3, ..., self_R)` 。
        - 如果 `dim` 为tuple(int)或list(int)，值为(2, 3)，并且 `keepdim` 为 ``False``，
          则输出的shape为 :math:`(self_1, self_4, ..., self_R)` 。

    异常：
        - **TypeError** - `keepdim` 不是bool类型。
        - **TypeError** - `self` 的数据类型或 `dtype` 是complex类型。
        - **ValueError** - `dim` 不在[-rank(self), rank(self))。

    .. py:method:: mindspore.Tensor.nansum(axis=None, keepdims=False, *, dtype=None) -> Tensor
        :noindex:

    计算输入Tensor指定维度元素的和，将非数字(NaNs)处理为零。

    参数：
        - **axis** (Union[int, tuple(int)], 可选) - 求和的维度。假设 `self` 的秩为r，取值范围[-r, r)。默认值： ``None`` ，对Tensor中的所有元素求和。
        - **keepdims** (bool, 可选) - 输出Tensor是否保持维度。默认值： ``False`` ，不保留维度。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 输出Tensor的类型。默认值： ``None`` 。

    返回：
        Tensor，输入Tensor指定维度的元素和，将非数字(NaNs)处理为零。

        - 如果 `axis` 为 ``None``，且 `keepdims` 为 ``False``，则输出一个零维Tensor，表示输入Tensor中所有元素的和。
        - 如果 `axis` 为int，值为2，并且 `keepdims` 为 ``False``，则输出的shape为： :math:`(self_1, self_3, ..., self_R)` 。
        - 如果 `axis` 为tuple(int)或list(int)，值为(2, 3)，并且 `keepdims` 为 ``False``，则输出的shape为 :math:`(self_1, self_4, ..., self_R)` 。

    异常：
        - **TypeError** - `keepdims` 不是bool类型。
        - **TypeError** - `self` 的数据类型或 `dtype` 是complex类型。
        - **ValueError** - `axis` 的范围不在[-r, r)，r表示 `self` 的秩。
