mindspore.ops.Xlogy
====================

.. py:class:: mindspore.ops.Xlogy

    计算第一个输入Tensor乘以第二个输入Tensor的对数。当 `input` 为零时，则返回零。

    .. math::

        out_i = input_{i}\ln{other_{i}}

    `input` 和 `other` 的输入遵循隐式类型转换规则，使数据类型一致。

    输入：
        - **input** (Tensor) - 第一个输入tensor，数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 。
        - **other** (Tensor) - 第二个输入tensor，数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 。

    输出：
        - **y** (Tensor) - shape是 `input` 和 `other` 广播后的shape，数据类型为两个输入中精度较高或数值较高的类型。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `other` 不是Tensor。
        - **ValueError** - 如果 `input` 和 `other` 的shape不可广播。