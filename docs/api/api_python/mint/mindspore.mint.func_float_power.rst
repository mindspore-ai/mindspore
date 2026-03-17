mindspore.mint.float_power
==========================

.. py:function:: mindspore.mint.float_power(input, exponent)

    使用双精度计算 `input` 中每个元素的 `exponent` 次幂，并返回一个float64类型的Tensor。

    .. math::

        out_{i} = input_{i} ^ {exponent_{i}}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        和 `ops.pow` 不同，该函数总是使用双精度进行计算，而 `ops.pow` 的计算精度则取决于类型提升结果。
        目前该函数不支持复数计算。
        由于在Ascend设备上float64类型计算性能比不上其他数据类型，因此建议该函数有且仅有在追求双精度且对性能要求不高的场景使用，否则 `ops.pow` 会是更好的选项。

    参数：
        - **input** (Union[Tensor, Number]) - 第一个输入，为Tensor或数值型数据类型。
        - **exponent** (Union[Tensor, Number]) - 第二个输入。如果 `input` 是Tensor，`exponent` 可以是Number或Tensor。否则，`exponent` 必须是Tensor。

    返回：
        Tensor，输出的shape与广播后的shape相同，返回的数据类型为mindspore.float64。

    异常：
        - **TypeError** - `input` 和 `exponent` 都不是Tensor。
        - **TypeError** - `input` 或 `exponent` 既不是Tensor也不是数值，或者其数据中包含了复数。
        - **ValueError** - `input` 和 `exponent` 具有不同的shape并且无法互相广播。
