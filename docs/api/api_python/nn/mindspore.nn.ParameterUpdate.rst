mindspore.nn.ParameterUpdate
=========================================

.. py:class:: mindspore.nn.ParameterUpdate(param)

    更新参数的Cell。

    使用输入的 `Tensor` 值更新 `param` 的值。

    参数：
        - **param** (Parameter) - 输入的参数。

    输入：
        - **x** (Tensor) - shape和type与 `param` 相同的Tensor。

    输出：
        Tensor，更新后的值。

    异常：
        - **KeyError** - 指定名称的参数不存在。
