mindspore.ops.pinv
=========================

.. py:function:: mindspore.ops.pinv(x, *, atol=None, rtol=None, hermitian=False)

    计算矩阵的（Moore-Penrose）伪逆。

    本函数通过SVD计算。如果 math:`x=U*S*V^{T}` ，则x的伪逆为 :math:`x^{T}=V*S^{+}*U^{T}` ，:math:`S^{+}` 为对S的对角线上的每个非零元素取倒数，零保留在原位。

    支持批量矩阵，若x是批量矩阵，当atol或rtol为float时，则输出具有相同的批量维度。
    若atol或rtol为Tensor，则其shape必须可广播到 `x.svd() <https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.svd.html>`_ 返回的奇异值的shape。
    若x.shape为 :math:`(B, M, N)` ，atol或rtol的shape为 :math:`(K, B)` ，则输出shape为 :math:`(K, B, N, M)` 。

    当hermitian为True时，暂时仅支持实数域，默认输入x为实对称矩阵，因此不会在内部检查x，并且在计算中仅使用下三角部分。
    当x的奇异值（或特征值范数，hermitian=True）小于阈值（ :math:`max(atol, \sigma \cdot rtol)` ， :math:`\sigma` 为最大奇异值或特征值）时，将其置为零，且在计算中不使用。

    若rtol和atol均未指定并且x的shape(M, N)，则rtol设置为 :math:`rtol=max(M, N)\varepsilon` ， :math:`\varepsilon` 为x.dtype的 `eps值 <https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Eps.html>`_ 。
    若rtol未指定且atol指定值大于零，则rtol设置为零。

    .. note::
        该函数在内部使用 `svd <https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.svd.html>`_
        （或 `eigh <https://www.mindspore.cn/docs/zh-CN/master/api_python/scipy/mindspore.scipy.linalg.eigh.html>`_ ,hermitian=True），
        因此与这些函数具有相同问题，详细信息请参阅svd()和eigh()中的警告。

    参数：
        - **x** (Tensor) - 要计算的矩阵。支持数据类型为float32或float64。shape为 :math:`(*, M, N)` ，其中*为零或多个批量维度。

          - hermitian为True时，暂不支持多个批量维度。

    关键字参数：
        - **atol** (float, Tensor) - 绝对公差值。默认值：None。
        - **rtol** (float, Tensor) - 相对公差值。默认值：None。
        - **hermitian** (bool) - 为True时求解x为实对称的矩阵。默认值：False。

    输出：
        - **output** (Tensor): 类型与输入相同。shape为 :math:`(*, N, M)` ，其中*为零或多个批量维度。

    异常：
        - **TypeError** - `hermitian` 不是bool。
        - **TypeError** - `x` 不是Tensor。
        - **ValueError** - `x` 的维度小于2。