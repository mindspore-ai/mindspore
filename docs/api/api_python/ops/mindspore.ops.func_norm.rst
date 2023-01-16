mindspore.ops.norm
==================

.. py:function:: mindspore.ops.norm(A, ord=None, dim=None, keepdim=False, *, dtype=None)

    返回给定Tensor的矩阵范数或向量范数。

    该函数计算向量范数或者矩阵范数的规则如下：

    - 如果 `dim` 是一个整型，将会计算向量范数。

    - 如果 `dim` 是一个2-tuple，将会计算矩阵范数。

    - 如果 `dim` 为None且 `ord` 为None，A将会被展平为1D并计算向量的2-范数。

    - 如果 `dim` 为None且 `ord` 不为None，A必须为1D或者2D。

    `ord` 为norm的计算模式。支持下列norm模式。

    ======================     =========================  ========================================================
    `ord`                      矩阵范数                     向量范数
    ======================     =========================  ========================================================
    `None` (默认值)             Frobenius norm             `2`-norm (参考最下方公式)
    `'fro'`                    Frobenius norm             不支持
    `'nuc'`                    nuclear norm               不支持
    `inf`                      `max(sum(abs(x), dim=1))`  `max(abs(x))`
    `-inf`                     `min(sum(abs(x), dim=1))`  `min(abs(x))`
    `0`                        不支持                      `sum(x != 0)`
    `1`                        `max(sum(abs(x), dim=0))`  参考下方公式
    `-1`                       `min(sum(abs(x), dim=0))`  参考下方公式
    `2`                        最大奇异值                  参考下方公式
    `-2`                       最小奇异值                  参考下方公式
    其余int或float值            不支持                     `sum(abs(x)^{ord})^{(1 / ord)}`
    ======================     =========================  ========================================================

    参数：
        - **A** (Tensor) - shape为 (*, n) 或者 (*, m, n)的Tensor，其中*是零个或多个batch维度。
        - **ord** (Union[int, float, inf, -inf, 'fro', 'nuc'], 可选) - norm的模式。行为参考上表。默认值：None。
        - **dim** (Union[int, Tuple(int)], 可选) - 计算向量范数或矩阵范数的维度。有关 `dim` = `None` 时的行为，请参见上文。默认值：None。
        - **keepdim** (bool) - 输出Tensor是否保留原有的维度。默认值：False。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 如果指定，则在执行之前将输入Tensor转换为dtype类型，返回的Tensor类型也将为dtype。默认值：None。

    返回：
        实值Tensor。

    异常：
        - **ValueError** - `dim` 超出范围。
        - **TypeError** - `dim` 既不是int也不是由int组成的tuple。
        - **TypeError** - `A` 是一个向量并且 `ord` 是str类型。
        - **ValueError** - `A` 是一个矩阵并且 `ord` 不是有效的取值。
        - **ValueError** - `A` 是一个矩阵并且 `ord` 为一个整型但是取值不为[1, -1, 2, -2]之一。
        - **ValueError** - `dim` 的两个元素在标准化过后取值相同。
        - **ValueError** - `dim` 的任意元素超出索引。
