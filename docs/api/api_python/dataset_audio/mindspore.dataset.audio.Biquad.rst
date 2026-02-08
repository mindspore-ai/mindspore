mindspore.dataset.audio.Biquad
==============================

.. py:class:: mindspore.dataset.audio.Biquad(b0, b1, b2, a0, a1, a2)

    给音频波形施加双二阶滤波器。

    具体的数学公式与参数详见 `数字双二阶滤波器 <https://zh.m.wikipedia.org/wiki/%E6%95%B0%E5%AD%97%E6%BB%A4%E6%B3%A2%E5%99%A8>`_ 。

    参数：
        - **b0** (float) - 当前输入的分子系数，x[n]。
        - **b1** (float) - 一个时间间隔前输入的分子系数，x[n-1]。
        - **b2** (float) - 两个时间间隔前输入的分子系数，x[n-2]。
        - **a0** (float) - 当前输出y[n]的分母系数，该值不能为0，通常为 ``1`` 。
        - **a1** (float) - 当前输出y[n-1]的分母系数。
        - **a2** (float) - 当前输出y[n-2]的分母系数。

    异常：
        - **TypeError** - 如果 `b0` 不是float类型。
        - **TypeError** - 如果 `b1` 不是float类型。
        - **TypeError** - 如果 `b2` 不是float类型。
        - **TypeError** - 如果 `a0` 不是float类型。
        - **TypeError** - 如果 `a1` 不是float类型。
        - **TypeError** - 如果 `a2` 不是float类型。
        - **ValueError** - 如果 `a0` 为0。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
