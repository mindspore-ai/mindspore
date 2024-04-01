mindspore.ops.silent_check.ASDBase
==================================

.. py:class:: mindspore.ops.silent_check.ASDBase(op)

    ASDBase 是 Python 中具有精度敏感检测特性的算子的基类。

    参数：
        - **op** (Primitive) - 需要增加精度敏感检测特性的原始算子。

    .. py:method:: generate_params()

        生成支持精度敏感检测的参数。

        返回：
            包含4个元素的元组。
            派生类通过调用此函数初始化精度敏感检测所需的参数。
