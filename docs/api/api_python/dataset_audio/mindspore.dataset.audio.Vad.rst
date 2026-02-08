mindspore.dataset.audio.Vad
===========================

.. py:class:: mindspore.dataset.audio.Vad(sample_rate, trigger_level=7.0, trigger_time=0.25, search_time=1.0, allowed_gap=0.25, pre_trigger_time=0.0, boot_time=0.35, noise_up_time=0.1, noise_down_time=0.01, noise_reduction_amount=1.35, measure_freq=20.0, measure_duration=None, measure_smooth_time=0.4, hp_filter_freq=50.0, lp_filter_freq=6000.0, hp_lifter_freq=150.0, lp_lifter_freq=2000.0)

    语音活动检测器。

    用于修剪语音记录末尾沉默或安静的背景声音。

    接口实现方式类似于 `SoX库 <https://sourceforge.net/projects/sox/>`_ 。

    参数：
        - **sample_rate** (int) - 音频信号的采样频率（单位：Hz）。
        - **trigger_level** (float, 可选) - 触发音频活动检测的测量级别。默认值： ``7.0`` 。
        - **trigger_time** (float, 可选) - 时间常数（以秒为单位），用于帮助忽略短促的爆破音。默认值： ``0.25`` 。
        - **search_time** (float, 可选) - 在检测到的触发点之前，搜索更安静、更短促爆破音的时长（以秒为单位）。默认值： ``1.0`` 。
        - **allowed_gap** (float, 可选) - 在检测到的触发点之前，允许的更安静、更短促爆破音之间的时间间隔（以秒为单位）。默认值： ``0.25`` 。
        - **pre_trigger_time** (float, 可选) - 在触发点和任意找到的更安静、更短促爆破音之前，要保留的音频时长（以秒为单位）。默认值： ``0.0`` 。
        - **boot_time** (float, 可选) - 初始噪声的评估时间（以秒为单位）。默认值： ``0.35`` 。
        - **noise_up_time** (float, 可选) - 用于自适应噪声评估的时间常量（以秒为单位），指定噪声等级上升时的时间。默认值： ``0.1`` 。
        - **noise_down_time** (float, 可选) - 用于自适应噪声评估的时间常量（以秒为单位），指定噪声等级下降时的时间。默认值： ``0.01`` 。
        - **noise_reduction_amount** (float, 可选) - 检测算法中使用的降噪量。默认值： ``1.35`` 。
        - **measure_freq** (float, 可选) - 算法处理、检测的频率（单位：Hz）。默认值： ``20.0`` 。
        - **measure_duration** (float, 可选) - 检测持续的时间（以秒为单位）。默认值： ``None`` ，将使用两倍测量周期的时长。
        - **measure_smooth_time** (float, 可选) - 用于平滑频谱测量的时间常数（以秒为单位）。默认值： ``0.4`` 。
        - **hp_filter_freq** (float, 可选) - 施加到检测器算法输入上的高通滤波器的截止频率（单位：Hz）。默认值： ``50.0`` 。
        - **lp_filter_freq** (float, 可选) - 施加到检测器算法输入上的低通滤波器的截止频率（单位：Hz）。默认值： ``6000.0`` 。
        - **hp_lifter_freq** (float, 可选) - 施加到检测器算法输入上的高通提升器的截止频率（单位：Hz）。默认值： ``150.0`` 。
        - **lp_lifter_freq** (float, 可选) - 施加到检测器算法输入上的低通提升器的截止频率（单位：Hz）。默认值： ``2000.0`` 。

    异常：
        - **TypeError** - 当 `sample_rate` 的类型不为int。
        - **ValueError** - 当 `sample_rate` 不为正数。
        - **TypeError** - 当 `trigger_level` 的类型不为float。
        - **TypeError** - 当 `trigger_time` 的类型不为float。
        - **ValueError** - 当 `trigger_time` 为负数。
        - **TypeError** - 当 `search_time` 的类型不为float。
        - **ValueError** - 当 `search_time` 为负数。
        - **TypeError** - 当 `allowed_gap` 的类型不为float。
        - **ValueError** - 当 `allowed_gap` 为负数。
        - **TypeError** - 当 `pre_trigger_time` 的类型不为float。
        - **ValueError** - 当 `pre_trigger_time` 为负数。
        - **TypeError** - 当 `boot_time` 的类型不为float。
        - **ValueError** - 当 `boot_time` 为负数。
        - **TypeError** - 当 `noise_up_time` 的类型不为float。
        - **ValueError** - 当 `noise_up_time` 为负数。
        - **TypeError** - 当 `noise_down_time` 的类型不为float。
        - **ValueError** - 当 `noise_down_time` 为负数。
        - **ValueError** - 当 `noise_up_time` 小于 `noise_down_time` 。
        - **TypeError** - 当 `noise_reduction_amount` 的类型不为float。
        - **ValueError** - 当 `noise_reduction_amount` 为负数。
        - **TypeError** - 当 `measure_freq` 的类型不为float。
        - **ValueError** - 当 `measure_freq` 不为正数。
        - **TypeError** - 当 `measure_duration` 的类型不为float。
        - **ValueError** - 当 `measure_duration` 为负数。
        - **TypeError** - 当 `measure_smooth_time` 的类型不为float。
        - **ValueError** - 当 `measure_smooth_time` 为负数。
        - **TypeError** - 当 `hp_filter_freq` 的类型不为float。
        - **ValueError** - 当 `hp_filter_freq` 不为正数。
        - **TypeError** - 当 `lp_filter_freq` 的类型不为float。
        - **ValueError** - 当 `lp_filter_freq` 不为正数。
        - **TypeError** - 当 `hp_lifter_freq` 的类型不为float。
        - **ValueError** - 当 `hp_lifter_freq` 不为正数。
        - **TypeError** - 当 `lp_lifter_freq` 的类型不为float。
        - **ValueError** - 当 `lp_lifter_freq` 不为正数。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
