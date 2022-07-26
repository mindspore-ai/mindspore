mindspore.ops.PrimitiveWithCheck
================================

.. py:class:: mindspore.ops.PrimitiveWithCheck(name)

    PrimitiveWithCheck是Python中原语的基类，定义了检查算子输入参数的函数，但是使用了C++源码中注册的推理方法。

    可以重写三个方法来定义Primitive的检查逻辑： __check__()、check_shape()和check_dtype()。如果在Primitive中定义了__check__()，则__check__()的优先级最高。

    如果未定义__check__()，则可以定义check_shape()和check_dtype()来描述形状和类型的检查逻辑。可以定义infer_value()方法（如PrimitiveWithInfer），用于常量传播。

    参数：
        - **name** (str) - 当前Primitive的名称。

    .. py:method:: check_dtype(*args)

        检查输入参数的数据类型。

        参数：
            - **args** (:class:`mindspore.dtype`) - 输入的数据类型。

        返回：
            None。

    .. py:method:: check_shape(*args)

        检查输入参数的shape。

        .. note::
            Scalar的shape是一个空元组。

        参数：
            - **args** (tuple(int)) - 输入tensor的shape。

        返回：
            None。
