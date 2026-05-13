mindspore.numpy.isclose
=================================

.. py:function:: mindspore.numpy.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)

    返回一个bool类型的Tensor，用于表示两个Tensor在给定的容差范围内是否逐元素相等。
    容差值为正数，通常是非常小的数字。相对差异（ :math:`rtol * abs(b)` ）和绝对差异 ``atol`` 相加后，与 ``a`` 和 ``b`` 的绝对差异进行比较。

    .. note::
        对于有限值，isclose使用以下公式来测试两个浮点数是否相等： :math:`absolute(a - b) <= (atol + rtol * absolute(b))` 。在Ascend平台上，不支持包含inf或NaN的输入数组。

    参数：
        - **a** (Union[Tensor, list, tuple]) - 要比较的第一个输入Tensor。
        - **b** (Union[Tensor, list, tuple]) - 要比较的第二个输入Tensor。
        - **rtol** (numbers.Number，可选) - 相对容差参数（见说明）。默认值： ``1e-05`` 。
        - **atol** (numbers.Number，可选) - 绝对容差参数（见说明）。默认值： ``1e-08`` 。
        - **equal_nan** (bool，可选) - 是否将 ``NaN`` 视为相等。如果为True， `a` 中的 ``NaN`` 在输出Tensor中将被视为与 `b` 中的 ``NaN`` 相等。默认值： ``False`` 。

    返回：
        在给定容差范围内，表示 `a` 和 `b` 是否相等的 ``bool`` 类型Tensor。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。