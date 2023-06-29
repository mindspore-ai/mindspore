mindspore.ops.vector_norm
=========================

.. py:function:: mindspore.ops.vector_norm(x, ord=2, axis=None, keepdim=False, *, dtype=None)

    返回给定Tensor在指定维度上的向量范数。

    `ord` 为norm的计算模式。支持下列norm模式。

    =================   ==============================================
    `ord`                向量范数
    =================   ==============================================
    ``2`` (默认值)        ``2`` -norm (参考最下方公式)
    ``inf``              :math:`max(abs(x))`
    ``-inf``             :math:`min(abs(x))`
    ``0``                :math:`sum(x != 0)`
    其余int或float值       :math:`sum(abs(x)^{ord})^{(1 / ord)}`
    =================   ==============================================

    参数：
        - **A** (Tensor) - shape为 :math:`(*, n)` 的Tensor，其中*是零个或多个batch维度。
        - **ord** (Union[int, float, inf, -inf], 可选) - norm的模式。行为参考上表。默认值： ``2`` 。
        - **axis** (Union[int, Tuple(int)], 可选) - 计算向量范数的维度。默认值： ``None`` 。

          当 `axis` 是int或者tuple时，会在指定的维度上计算范数，而剩余的维度会被作为batch维度。

          当 `axis` 为None时，在计算范数之前，会将Tensor `x` 展平。

        - **keepdim** (bool) - 输出Tensor是否保留原有的维度。默认值： ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 如果设置此参数，则会在执行之前将 `x` 转换为指定的类型，返回的Tensor类型也将为指定类型 `dtype`。
          如果 `dtype` 为 ``None`` ，保持 `A` 的类型不变。默认值： ``None`` 。

    返回：
        Tensor，在指定维度 `axis` 上进行范数计算的结果，与输入 `x` 的数据类型相同。

    异常：
        - **TypeError** - `axis` 既不是int也不是由int组成的tuple。
        - **ValueError** - `ord` 不在[int, float, inf, -inf]中。
        - **ValueError** - `axis` 中的元素有重复。
        - **ValueError** - `axis` 中的任意元素超出了范围。
