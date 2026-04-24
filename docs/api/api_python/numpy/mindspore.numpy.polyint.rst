mindspore.numpy.polyint
=======================

.. py:function:: mindspore.numpy.polyint(p, m=1, k=None)

    返回多项式的一个反导数(不定积分)。

    .. note::
        目前不支持NumPy对象poly1d。

    参数：
        - **p** (Union[int, float, bool, list, tuple, Tensor]) - 要积分的多项式。 一个表示多项式系数的序列。
        - **m** (int, 可选) - 默认值：1，反导数的阶数。
        - **k** (Union[int, list[int]], 可选) - 积分常数。按积分顺序给出：对应最高阶项的常数排在最前面。如果为None(默认)，所有常数都设为零。如果 ``m = 1`` ，可以给定为标量而非列表。

    返回：
        Tensor，表示反导数的新多项式。

    异常：
        - **ValueError** - 如果 `p` 的维数超过1。