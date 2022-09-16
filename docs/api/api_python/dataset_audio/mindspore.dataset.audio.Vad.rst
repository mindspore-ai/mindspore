mindspore.dataset.audio.Vad
===========================

.. py:class:: mindspore.dataset.audio.Vad(sample_rate, trigger_level=7.0, trigger_time=0.25, search_time=1.0, allowed_gap=0.25, pre_trigger_time=0.0, boot_time=0.35, noise_up_time=0.1, noise_down_time=0.01, noise_reduction_amount=1.35, measure_freq=20.0, measure_duration=None, measure_smooth_time=0.4, hp_filter_freq=50.0, lp_filter_freq=6000.0, hp_lifter_freq=150.0, lp_lifter_freq=2000.0)

    从录音结束后修剪无声背景声音。

    参数：
        - **sample_rate** (int) - 采样频率（单位：Hz），不能为零。
        - **trigger_level** (float, 可选) - 用于触发活动检测的测量级别，默认值：7.0。
        - **trigger_time** (float, 可选) - 用于帮助忽略短音的时间常数（以秒为单位，默认值：0.25。
        - **search_time** (float, 可选) - 在检测到的触发点之前搜索要包括的更安静/更短声音的音频量（以秒为单位），默认值：1.0。
        - **allowed_gap** (float, 可选) - 包括检测到的触发点之前较短/较短声音之间允许的间隙（以秒为单位），默认值：0.25。
        - **pre_trigger_time** (float, 可选) - 在触发点和任何找到的更安静/更短的声音突发之前，要保留的音频量（以秒为单位），默认值：0.0。
        - **boot_time** (float, 可选) - 初始噪声估计的时间，默认值：0.35。
        - **noise_up_time** (float, 可选) - 当噪音水平增加时，自适应噪音估计器使用的时间常数，默认值：0.1。
        - **noise_down_time** (float, 可选) - 当噪音水平降低时，自适应噪音估计器使用的时间常数，默认值：0.01。
        - **noise_reduction_amount** (float, 可选) - 检测算法中使用的降噪量，默认值：1.35。
        - **measure_freq** (float, 可选) - 算法处理的频率，默认值：20.0。
        - **measure_duration** (float, 可选) - 测量持续时间，默认值：None，使用测量周期的两倍。
        - **measure_smooth_time** (float, 可选) - 用于平滑光谱测量的时间常数，默认值：0.4。
        - **hp_filter_freq** (float, 可选) - 应用于检测器算法输入的高通滤波器的"Brick-wall"频率，默认值：50.0。
        - **lp_filter_freq** (float, 可选) - 应用于检测器算法输入的低通滤波器的"Brick-wall"频率，默认值：6000.0。
        - **hp_lifter_freq** (float, 可选) - 应用于检测器算法输入的高通升降机的"Brick-wall"频率，默认值：150.0。
        - **lp_lifter_freq** (float, 可选) - 应用于检测器算法输入的低通升降机的"Brick-wall"频率，默认值：20000.0。
