mindspore.dataset.audio.SlidingWindowCmn
========================================

.. py:class:: mindspore.dataset.audio.SlidingWindowCmn(cmn_window=600, min_cmn_window=100, center=False, norm_vars=False)

    对每个话语应用滑动窗口倒谱均值（和可选方差）归一化。

    参数：
        - **cmn_window** (int, 可选) - 用于运行平均CMN计算的帧中窗口，默认值：600。
        - **min_cmn_window** (int, 可选) - 解码开始时使用的最小CMN窗口（仅在开始时增加延迟）。
          仅在中心为False时适用，在中心为True时忽略，默认值：100。
        - **center** (bool, 可选) - 如果为True，则使用以当前帧为中心的窗口。如果为False，则窗口在左侧。默认值：False。
        - **norm_vars** (bool, 可选) - 如果为True，则将方差规范化为1。默认值：False。
  
