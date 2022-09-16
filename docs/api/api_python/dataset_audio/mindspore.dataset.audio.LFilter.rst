mindspore.dataset.audio.LFilter
===============================

.. py:class:: mindspore.dataset.audio.LFilter(a_coeffs, b_coeffs, clamp=True)

    为（...，time）形状的音频波形施加双极滤波器。

    参数：
        - **a_coeffs** (sequence) - (n_order + 1)维数差分方程的分母系数。较低的延迟系数是第一位的，例如[a0, a1, a2, ...]。
          大小必须与 `b_coeffs` 相同（根据需要填充0）。
        - **b_coeffs** (sequence) - (n_order + 1)维数差分方程的分子系数。较低的延迟系数是第一位的，例如[b0, b1, b2, ...]。
          大小必须与 `a_coeffs` 相同（根据需要填充0）。
        - **clamp** (bool, 可选) - 如果为True，则将输出信号截断在[-1, 1]范围内，默认值：True。
    
    异常：
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
