mindspore.dataset.audio.SlidingWindowCmn
========================================

.. py:class:: mindspore.dataset.audio.SlidingWindowCmn(cmn_window=600, min_cmn_window=100, center=False, norm_vars=False)

    对每个话语应用滑动窗口倒谱均值（和可选方差）归一化。

    参数：
        - **cmn_window** (int, 可选) - 用于运行平均CMN计算的帧中窗口。默认值： ``600`` 。
        - **min_cmn_window** (int, 可选) - 解码开始时使用的最小CMN窗口（仅在开始时增加延迟）。
          仅在 `center` 为 ``False`` 时适用，在 `center` 为 ``True`` 时忽略。默认值： ``100`` 。
        - **center** (bool, 可选) - 如果为 ``True`` ，则使用以当前帧为中心的窗口；如果为 ``False`` ，则窗口在左侧。默认值： ``False`` 。
        - **norm_vars** (bool, 可选) - 如果为 ``True`` ，则将方差规范化为1。默认值： ``False`` 。

    异常：
        - **TypeError** - 当 `cmn_window` 的类型不为int。
        - **ValueError** - 当 `cmn_window` 为负数。
        - **TypeError** - 当 `min_cmn_window` 的类型不为int。
        - **ValueError** - 当 `min_cmn_window` 为负数。
        - **TypeError** - 当 `center` 的类型不为bool。
        - **TypeError** - 当 `norm_vars` 的类型不为bool。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
