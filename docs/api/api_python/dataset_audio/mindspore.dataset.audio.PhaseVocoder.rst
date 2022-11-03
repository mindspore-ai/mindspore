mindspore.dataset.audio.PhaseVocoder
====================================

.. py:class:: mindspore.dataset.audio.PhaseVocoder(rate, phase_advance)

    对给定的STFT频谱，在不改变音高的情况下以一定比率进行加速。

    参数：
        - **rate** (float) - 加速比率。
        - **phase_advance** (numpy.ndarray) - 每个频段的预期相位提前量，shape为（freq, 1）。

    异常：
        - **TypeError** - 当 `rate` 的类型不为float。
        - **ValueError** - 当 `rate` 不为正数。
        - **TypeError** - 当 `phase_advance` 的类型不为 :class:`numpy.ndarray` 。
        - **RuntimeError** - 当输入音频的shape不为<..., freq, num_frame, complex=2>。
