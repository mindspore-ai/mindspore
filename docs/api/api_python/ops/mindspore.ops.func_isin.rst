mindspore.ops.isin
===================

.. py:function:: mindspore.ops.isin(elements, test_elements, *, assume_unique=False, invert=False)

    计算 `elements` 中的每个元素是否都在 `test_elements` 中，仅在 `elements` 上进行广播。

    参数：
        - **elements** (Union[int, float, bool, list, tuple, Tensor]) - 输入数组。
        - **test_elements** (Union[int, float, bool, list, tuple, Tensor]) - 用于测试 `input` 的每个值的值。
        - **assume_unique** (bool, 可选) - 是否假定 `test_elements` 中的每个元素均为唯一。如果为 ``True`` ，则假定 `test_elements` 包含元素均为唯一，这可以加快计算速度。默认值： ``False``。
        - **invert** (boole, 可选) - 如果为True，则返回的数组中的值将被反转，就像计算 `input` 不在 `test_elements` 中一样。默认值：``False``。

    返回：
        返回与 `elements` 相同shape的布尔Tensor，其中在 `test_elements` 中的元素为 ``True`` ，否则为 ``False`` 。
