mindspore.dataset.audio.Dither
==============================

.. py:class:: mindspore.dataset.audio.Dither(density_function=DensityFunction.TPDF, noise_shaping=False)

    通过消除非线性截断失真，来抖动增加存储在特定位深的音频的动态感知范围。

    参数：
        - **density_function** (:class:`mindspore.dataset.audio.DensityFunction` , 可选) - 连续随机变量的密度函数。
          可为DensityFunction.TPDF（三角概率密度函数）、DensityFunction.RPDF（矩形概率密度函数）
          或DensityFunction.GPDF（高斯概率密度函数）。默认值：DensityFunction.TPDF。
        - **noise_shaping** (bool, 可选) - 是否通过滤波操作，来消除频谱能量的量化误差。默认值：False。

    异常：
        - **TypeError** - 当 `density_function` 的类型不为 :class:`mindspore.dataset.audio.DensityFunction` 。
        - **TypeError** - 当 `noise_shaping` 的类型不为bool。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
