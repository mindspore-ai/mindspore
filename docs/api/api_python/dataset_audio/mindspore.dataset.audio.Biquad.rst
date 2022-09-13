mindspore.dataset.audio.Biquad
==============================

.. py:class:: mindspore.dataset.audio.Biquad(b0, b1, b2, a0, a1, a2)

    给音频波形施加双二阶滤波器。

    具体的数学公式与参数详见 `数字双二阶滤波器 <https://zh.m.wikipedia.org/wiki/%E6%95%B0%E5%AD%97%E6%BB%A4%E6%B3%A2%E5%99%A8>`_ 。

    参数：
        - **b0** (float) - 电流输入的分子系数，x[n]。
        - **b1** (float) - 一个时间间隔前输入的分子系数x[n-1]。
        - **b2** (float) - 两个时间间隔前输入的分子系数x[n-2]。
        - **a0** (float) - 电流输出y[n]的分母系数，该值不能为零，通常为1。
        - **a1** (float) - 电流输出y[n-1]的分母系数。
        - **a2** (float) - 电流输出y[n-2]的分母系数。
