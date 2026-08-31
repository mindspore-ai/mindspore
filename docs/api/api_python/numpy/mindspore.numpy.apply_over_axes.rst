mindspore.numpy.apply_over_axes
=================================

.. py:function:: mindspore.numpy.apply_over_axes(func, a, axes)

    在多个轴上重复应用 ``func`` 函数。 ``res = func(a, axis)`` ，其中 ``axis`` 是 ``axes`` 的第一个元素。函数返回的结果 ``res`` 具有与 ``a`` 相同的维度或少一个维度。如果 ``res`` 比 ``a`` 少一个维度，则在 ``axis`` 之前插入一个维度。然后将 ``res`` 作为 ``func`` 的第一个参数，对 ``axes`` 中的每个轴重复调用 ``func`` 。

    参数：
        - **func** (function) - 该函数必须有两个参数 ``a`` ， ``axis`` 。
        - **a** (Union[int, float, bool, list, tuple, Tensor]) - 输入的Tensor。
        - **axes** (Union[int, list, tuple]) - 应用 ``func`` 函数的轴。输入元素必须是整数

    返回：
        Tensor，维度与 ``a`` 的相同，但shape可以不同。这取决于 ``func`` 函数是否改变输出的shape。

    异常：
        - **TypeError** - 如果 ``a`` 不是类似数组的对象，或者 ``axes`` 参数的类型不是int型或int型的序列。
        - **ValueError** - 如果 ``axes`` 中的轴重复或超出索引范围。