mindspore.dataset.audio.PhaseVocoder
====================================

.. py:class:: mindspore.dataset.audio.PhaseVocoder(rate, phase_advance)

    给定STFT张量，在不修改速率系数的情况下加速音高。

    参数：
        - **rate** (float) - 加速系数。
        - **phase_advance** (numpy.ndarray) - 每个滤波器的预期相位前进，形状为（freq, 1）。
