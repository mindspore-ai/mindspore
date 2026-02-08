mindspore.ops.norm
==================

.. py:function:: mindspore.ops.norm(A, ord=None, dim=None, keepdim=False, *, dtype=None)

    计算tensor的矩阵范数或向量范数。

    `ord` 为norm的计算模式。支持下列norm模式。

    =================   ================================== ==============================================
    `ord`                矩阵范数                               向量范数
    =================   ================================== ==============================================
    `None` (默认值)      Frobenius norm                     `2`-norm (参考最下方公式)
    `'fro'`              Frobenius norm                     不支持
    `'nuc'`              nuclear norm                       不支持
    `inf`                :math:`max(sum(abs(x), dim=1))`    :math:`max(abs(x))`
    `-inf`               :math:`min(sum(abs(x), dim=1))`    :math:`min(abs(x))`
    `0`                  不支持                             :math:`sum(x != 0)`
    `1`                  :math:`max(sum(abs(x), dim=0))`    参考最下方公式
    `-1`                 :math:`min(sum(abs(x), dim=0))`    参考最下方公式
    `2`                  最大奇异值                         参考最下方公式
    `-2`                 最小奇异值                         参考最下方公式
    其余int或float值     不支持                             :math:`sum(abs(x)^{ord})^{(1 / ord)}`
    =================   ================================== ==============================================

    参数：
        - **A** (Tensor) - 输入tensor。
        - **ord** (Union[int, float, inf, -inf, 'fro', 'nuc'], 可选) - 指定要采用的范数类型。默认 ``None`` 。
        - **dim** (Union[int, Tuple(int)], 可选) - 指定计算维度。默认 ``None`` 。

          - 当 `dim` 为int时，计算向量范数。
          - 当 `dim` 为一个二元组时，计算矩阵范数。
          - 当 `dim` 为 ``None`` 且 `ord` 为 ``None`` ，展平 `A` 为一维tensor并计算向量的2-范数。
          - 当 `dim` 为 ``None`` 且 `ord` 不为 ``None`` ， `A` 必须为一维或者二维。

        - **keepdim** (bool) - 输出tensor是否保留维度。默认 ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定数据类型。如果设置此参数，则会在计算前将 `A` 转换为指定的类型。默认 ``None`` 。

    返回：
        Tensor

    .. note::
        - 当前暂不支持复数。
        - 大小值域输入情况，Ascend后端的计算结果可能出现精度误差。
