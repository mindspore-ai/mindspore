mindspore.ops.PrimitiveWithInfer
================================

.. py:class:: mindspore.ops.PrimitiveWithInfer(name)

    PrimitiveWithInfer是Python中的原语基类，在python中定义了跟踪推理的函数。

    可以重写四个方法来定义Primitive的推断逻辑：__infer__()、infer_shape()、infer_dtype()和infer_value()。如果在Primitive中定义了__infer__()，则__infer__()的优先级最高。

    如果未定义__infer__()，则可以定义infer_shape()和infer_dtype()来描述shape和类型的推断逻辑。infer_value()用于常量传播。

    **参数：**

    - **name** (str) - 当前Primitive的名称。

    .. py:method:: infer_dtype(*args)

        根据输入类型推断输出类型。

        **参数：**

        - **args** (:class:`mindspore.dtype`) - 输入的数据类型。

        **返回：**

        :class:`mindspore.dtype`，输出的数据类型。

    .. py:method:: infer_shape(*args)

        根据输入形状推断输出形状。

        .. note::
            Scalar的shape是一个空元组。

        **参数：**

        - **args** (tuple(int)) - 输入tensor的shape。

        **返回：**

        `tuple(int)`，输出tensor的shape。

    .. py:method:: infer_value(*args)

        根据编译时的输入值推断输出值。

        **参数：**

        - **args** (Any) - 输入的值。

        **返回：**

        输出的值。如果编译时无法推断该值，返回 `None` 。
