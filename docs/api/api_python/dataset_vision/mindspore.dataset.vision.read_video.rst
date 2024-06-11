mindspore.dataset.vision.read_video
===================================

.. py:function:: mindspore.dataset.vision.read_video(filename, start_pts=0, end_pts=None, pts_unit="pts")

    从视频文件中读取视频、音频、元数据。

    支持的文件类型有AVI、H264、H265、MOV、MP4和WMV。

    参数：
        - **filename** (str) - 待读取视频文件路径。
        - **start_pts** (Union[float, Fraction, int], 可选) - 视频的开始时间戳。默认值: 0。
        - **end_pts** (Union[float, Fraction, int], 可选) - 视频的结束时间戳。默认值: None，对应2147483647。
        - **pts_unit** (str, 可选) - 时间戳的单位，它可以是["pts", "sec"]中的任何一个。默认值: "pts"。

    返回：
        - numpy.ndarray, 四维 uint8 视频数据。格式为 [T, H, W, C]。“T”是帧数，“H”是高度，“W”是宽度，“C”是RGB的通道。
        - numpy.ndarray, 二维音频数据。格式为 [C, L]。“C”是通道数，“L”是一个通道中数据的点数。
        - dict, 视频和音频的元数据。它包含float类型的video_fps数据和int类型的audio_fps数据。

    异常：
        - **TypeError** - 如果 `filename` 不是str类型。
        - **TypeError** - 如果 `start_pts` 的类型不是Union[float, Fraction, int]类型。
        - **TypeError** - 如果 `end_pts` 的类型不是Union[float, Fraction, int]类型。
        - **TypeError** - 如果 `pts_unit` 不是str类型。
        - **RuntimeError** - 如果 `filename` 不存在，或不是普通文件，或由于格式等原因无法正常读取。
        - **ValueError** - 如果 `start_pts` 小于 0。
        - **ValueError** - 如果 `end_pts` 小于 `start_pts`。
        - **ValueError** - 如果 `pts_unit` 不在 ["pts", "sec"] 中。
