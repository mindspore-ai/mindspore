mindspore.ops.ExtractGlimpse
=============================

.. py:class:: mindspore.ops.ExtractGlimpse(centered=True, normalized=True, uniform_noise=True, noise="uniform")

    从输入图像Tensor中提取glimpses（通常为矩形子区域），并作为窗口返回。

    .. note::
        如果提取的窗口和输入图像仅部分重叠，用随机噪声对非重叠部分进行填充。

    参数：
        - **centered** (bool，可选) - 可选的bool，指示偏移坐标是否相对于图像居中，如果为 ``True`` ，表示
          (0,0)偏移是相对于输入图像的中心的；如果为 ``False`` ，则(0,0)偏移量对应于输入图像的左上角。默认为 ``True`` 。
        - **normalized** (bool，可选) - 可选的bool，指示偏移坐标是否归一化。默认为 ``True`` 。
        - **uniform_noise** (bool，可选) - 可选的bool，指示是否应该使用均匀分布（即高斯分布）生成噪声。默认为 ``True`` 。
        - **noise** (str，可选) - 填充的噪声。窗口由输入大小和偏移决定，如果窗口与输入部分没有重叠，则填充随机噪声。其值可以为： ``"uniform"`` 、 ``"gaussian"`` 和 ``"zero"`` 。默认值： ``"uniform"`` 。
          
          - 当 `noise` 为 ``"uniform"`` 或者 ``"gaussian"`` ，其填充结果是变量。
          - 当 `noise` 为 ``"zero"`` ，则 `uniform_noise` 必须为 ``False`` ，这样填充的噪声才是0，保证结果的正确。
          - 当 `uniform_noise` 为 ``True`` ， `noise` 仅可以为 ``"uniform"`` 。当 `uniform_noise` 为 ``False`` ， `noise` 可以为 ``"uniform"`` 、 ``"gaussian"`` 或 ``"zero"`` 。

    输入：
        - **x** (Tensor) - 一个 `4-D` 的Tensor，shape为 :math:`(batch\_size, height, width, channels)` ，dtype为float32。
        - **size** (Tensor) - 一个包含2个元素的 `1-D` Tensor，包含了提取glimpses的大小。
          `glimpse` 的高度必须首先指定，然后是其宽度，数据类型为int32，其大小必须大于0。
        - **offsets** (Tensor) - 一个 `2-D` 的Tensor，shape为 :math:`(batch\_size, 2)`，包含了每个窗口中心点的y、x位置，数据类型为float32。

    输出：
        一个 `4-D` 的Tensor，shape为 :math:`(batch\_size, glimpse\_height, glimpse\_width, channels)` ，数据类型为float32。

    异常：
        - **TypeError** - 如果 `centered` 不是一个bool。
        - **TypeError** - 如果 `normalize` 不是一个bool。
        - **TypeError** - 如果 `uniform_noise` 不是一个bool。
        - **ValueError** - 如果 `noise` 不是 "uniform" 、 "gaussian" 或者 "zero"。
        - **ValueError** - 如果 `size` 的值不是常数。
        - **ValueError** - 如果输入 `x` 和 `offsets` 的batch_size不一致。
        - **ValueError** - 如果 `offsets[1]` 不是2。
        - **ValueError** - 如果输入 `x` 不是一个Tensor。
