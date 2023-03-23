mindspore.dataset.vision.read_image
===================================

.. py:function:: mindspore.dataset.vision.read_image(filename, mode=ImageReadMode.UNCHANGED)

    读取图像文件并解码为3通道RGB彩色数据或灰度数据。
    支持的文件类型有JPEG、PNG、BMP和TIFF。

    参数：
        - **filename** (str) - 待读取图像文件路径。
        - **mode** (:class:`mindspore.dataset.vision.ImageReadMode` , 可选) - 图像读取模式。它可以是 [ImageReadMode.UNCHANGED、ImageReadMode.GRAYSCALE、ImageReadMode.COLOR] 
          中的任何一个。默认值：ImageReadMode.UNCHANGED。

          - **ImageReadMode.UNCHANGED** - 按照图像原始格式读取。
          - **ImageReadMode.GRAYSCALE** - 读取并转为单通道灰度数据。
          - **ImageReadMode.COLOR** - 读取并换为3通道RGB彩色数据。

    返回：
        - numpy.ndarray, 三维uint8类型数据，shape为（H, W, C）。

    异常：
        - **TypeError** - 如果 `filename` 不是str类型。
        - **TypeError** - 如果 `mode` 不是 :class:`mindspore.dataset.vision.ImageReadMode` 类型。
        - **RuntimeError** - 如果 `filename` 不存在或不是普通文件或由于格式等原因无法正常读取。
