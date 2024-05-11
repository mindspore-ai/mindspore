mindspore.dataset.vision.read_video_timestamps
==============================================

.. py:function:: mindspore.dataset.vision.read_video_timestamps(filename, pts_unit="pts")

    读取视频文件的时间戳和帧率。
    支持的文件类型有AVI、H264、H265、MOV、MP4和WMV。

    参数：
        - **filename** (str) - 待读取视频文件路径。
        - **pts_unit** (str, 可选) - 时间戳的单位，它可以是["pts", "sec"]中的任何一个。默认值: "pts"。

    返回：
        - list, 当 `pts_unit` 为"pts"时返回list[int]，当 `pts_unit` 为"sec"时返回list[float]。
        - float, 视频的每秒帧数。

    异常：
        - **TypeError** - 如果 `filename` 不是str类型。
        - **TypeError** - 如果 `pts_unit` 不是str类型。
        - **RuntimeError** - 如果 `filename` 不存在，或不是普通文件，或由于格式等原因无法正常读取。
        - **RuntimeError** - 如果 `pts_unit` 不在 ["pts", "sec"] 中。
