mindspore.dataset.audio.Dither
==============================

.. py:class:: mindspore.dataset.audio.Dither(density_function=DensityFunction.TPDF, noise_shaping=False)

    通过消除非线性截断失真，增加存储在特定位深度的音频的感知动态范围。

    参数：
        - **density_function** (DensityFunction, 可选) - 连续随机变量的密度函数。
          可以是DensityFunction.TPDF（三角形概率密度函数）、DensityFunction.RPDF（矩形概率密度函数）
          或DensityFunction.GPDF（高斯概率密度函数）之一。默认值：DensityFunction.TPDF。
        - **noise_shaping** (bool, 可选) - 是否塑造量化误差的光谱能量。默认值：False。
