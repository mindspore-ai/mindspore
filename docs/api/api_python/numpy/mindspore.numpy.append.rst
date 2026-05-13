mindspore.numpy.append
=================================

.. py:function:: mindspore.numpy.append(arr, values, axis=None)

    将值添加到Tensor的末尾。

    参数：
        - **arr** (Tensor) - 需要被添加 ``values`` 的数组。
        - **values** (Tensor) - 添加到数组 ``arr`` 中的值。该输入的形状必须正确（与 ``arr`` 的形状相同，不包括 ``axis`` ）。如果未指定 ``axis`` ，则值可以是任何形状，并且在使用前将被展平。
        - **axis** (int, 可选) - 所要附加的 ``values`` 的轴。如果未给定 ``axis`` ，则 ``arr`` 和 ``values`` 在使用前都会被展平，默认值： ``None`` 。

    返回：
        Tensor，添加值到指定 ``axis`` 上的Tensor。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果 ``axis`` 超过了 ``arr.ndim`` 。