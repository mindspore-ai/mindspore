mindspore.ops.silent_check.ASDBase
==================================

.. py:class:: mindspore.ops.silent_check.ASDBase(cls, *args, **kwargs)

    ASDBase 是 Python 中具有特征值检测特性的算子的基类。

    参数：
        - **cls** (Primitive) - 需要增加特征值检测特性的原始算子。
        - **args** (tuple) - 传递给原始运算符的可变参数元组。
        - **kwargs** (dict) - 传递给原始运算符的可变参数字典。

    .. py:method:: generate_params()

        生成支持特征值检测的参数。

        返回：
            包含4个元素的元组。
            派生类通过调用此函数初始化特征值检测所需的参数。
