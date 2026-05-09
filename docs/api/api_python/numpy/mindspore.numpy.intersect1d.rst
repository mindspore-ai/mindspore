mindspore.numpy.intersect1d
=================================

.. py:function:: mindspore.numpy.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)

    查找两个Tensor的交集。返回两个输入Tensor中都存在的、已排序去重的值。

    参数：
        - **ar1** (Union[int, float, bool, list, tuple, Tensor]) - 输入Tensor。
        - **ar2** (Union[int, float, bool, list, tuple, Tensor]) - 输入Tensor。
        - **assume_unique** (bool) - 如果为 ``True`` ，则假设输入Tensor没有重复的元素，这可以加快计算速度。若为 ``True`` 但 `ar1` 或 `ar2` 不唯一，可能会导致结果不正确或索引超出范围。默认值： ``False`` 。
        - **return_indices** (bool) -  如果为 ``True`` ，返回与交集对应的索引。如果值出现多次，则使用第一次出现的索引。默认值： ``False`` 。

    返回：
        Tensor或Tensor的tuple。若 `return_indices` 为 ``False`` ，则返回交集Tensor；否则返回Tensor的tuple。

    异常：
        - **TypeError** - 如果输入的 `ar1` 或 `ar2` 不是类似数组的对象。
        - **TypeError** - 如果 `assume_unique` 或 `return_indices` 不是bool类型。
