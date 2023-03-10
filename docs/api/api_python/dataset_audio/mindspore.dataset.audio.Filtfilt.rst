mindspore.dataset.audio.Filtfilt
================================

.. py:class:: mindspore.dataset.audio.Filtfilt(a_coeffs, b_coeffs, clamp=True)

    对音频波形施加正反向IIR滤波。

    参数：
        - **a_coeffs** (Sequence[float]) - 不同维度的差分方程分母系数。低维的延迟系数在前，例如[a0, a1, a2, ...]。
          维度必须与 `b_coeffs` 相同（根据需要填充0值）。
        - **b_coeffs** (Sequence[float]) - 不同维度的差分方程分子系数。低维的延迟系数在前，例如[b0, b1, b2, ...]。
          维度必须与 `a_coeffs` 相同（根据需要填充0值）。
        - **clamp** (bool, 可选) - 如果为True，将输出信号截断在[-1, 1]范围内。默认值：True。

    异常：
        - **TypeError** - 当 `a_coeffs` 的类型不为Sequence[float]。
        - **TypeError** - 当 `b_coeffs` 的类型不为Sequence[float]。
        - **ValueError** - 当 `a_coeffs` 与 `b_coeffs` 维度不同。
        - **TypeError** - 当 `clamp` 的类型不为bool。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
