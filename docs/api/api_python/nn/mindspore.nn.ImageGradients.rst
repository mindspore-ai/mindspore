mindspore.nn.ImageGradients
============================

.. py:class:: mindspore.nn.ImageGradients

    计算每个颜色通道的图像渐变，返回为两个Tensor，分别表示高和宽方向上的变化率。

    假设图像shape为 :math:`h*w` ，则沿高和宽的梯度分别为 :math:`dy` 和 :math:`dx` 。

    .. math::
        dy[i] = \begin{cases} image[i+1, :]-image[i, :], &if\ 0<=i<h-1 \cr
        0, &if\ i==h-1\end{cases}

        dx[i] = \begin{cases} image[:, i+1]-image[:, i], &if\ 0<=i<w-1 \cr
        0, &if\ i==w-1\end{cases}

    **输入：**

    - **images** (Tensor) - 输入图像数据，格式为'NCHW'。

    **输出：**

    - **dy** (Tensor) - 垂直方向的图像梯度，数据类型和shape与输入相同。
    - **dx** (Tensor) - 水平方向的图像梯度，数据类型和shape与输入相同。

    **异常：**

    - **ValueError** - `images` 的shape长度不等于4。