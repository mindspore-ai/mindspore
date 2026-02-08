mindspore.dataset.audio.DetectPitchFrequency
============================================

.. py:class:: mindspore.dataset.audio.DetectPitchFrequency(sample_rate, frame_time=0.01, win_length=30, freq_low=85, freq_high=3400)

    检测音调频率。
    基于归一化互相关函数和中位平滑来实现。

    参数：
        - **sample_rate** (int) - 波形的采样频率（单位：Hz），该值不能为零，如： ``44100`` 。
        - **frame_time** (float, 可选) - 帧的持续时间，该值必须大于零。默认值： ``0.01`` 。
        - **win_length** (int, 可选) - 中位平滑的窗口长度（以帧数为单位），该值必须大于零。默认值： ``30`` 。
        - **freq_low** (int, 可选) - 可检测的最低频率（单位：Hz），该值必须大于零。默认值： ``85`` 。
        - **freq_high** (int, 可选) - 可检测的最高频率（单位：Hz），该值必须大于零。默认值： ``3400`` 。

    异常：
        - **TypeError** - 如果 `sample_rate` 不是int类型。
        - **ValueError** - 如果 `sample_rate` 为0。
        - **TypeError** - 如果 `frame_time` 不是float类型。
        - **ValueError** - 如果 `frame_time` 不为正数。
        - **TypeError** - 如果 `win_length` 不是int类型。
        - **ValueError** - 如果 `win_length` 不为正数。
        - **TypeError** - 如果 `freq_low` 不是int类型。
        - **ValueError** - 如果 `freq_low` 不为正数。
        - **TypeError** - 如果 `freq_high` 不是int类型。
        - **ValueError** - 如果 `freq_high` 不为正数。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
