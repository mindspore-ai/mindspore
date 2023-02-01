mindspore.nn.PoissonNLLLoss
============================

.. py:class:: mindspore.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8, reduction='mean')

    计算泊松负对数似然损失。

    损失为：

    .. math::
        \mathcal{L}_{D} = \sum_{i = 0}^{|D|}\left( x_{i} - y_{i}\ln x_{i} + \ln{y_{i}!} \right)

    其中 :math:`\mathcal{L}_{D}` 为损失值， :math:`y_{i}` 为 `target` ， :math:`x_{i}` 为 `x` 。

    如果 `log_input` 为True，使用 :math:`e^{x_{i}} - y_{i} x_{i}` 而不是 :math:`x_{i} - y_{i}\ln x_{i}` 进行计算。
    计算对数时，`x` 的下界设置为`eps`，以避免数值误差。
    如果 `full` 为False，则最后一项：:math:`\ln{y_{i}!}` 将被省略。否则，最后一项将使用斯特林公式近似：

    .. math::
        n! \approx \sqrt{2\pi n}\left( \frac{n}{e} \right)^{n}

    .. note::
        在Ascend下计算负数的对数或大正数的指数将具有与GPU和CPU下不同的返回值和结果范围。

    参数：
        - **log_input** (bool，可选) - 是否使用对数输入。默认值：True。
        - **full** (bool，可选) - 是否在损失计算中包括斯特林近似项。默认值：False。
        - **eps** (float，可选) - 算对数时 `x` 的下界。默认值：1e-8。
        - **reduction** (str，可选) - 指定要应用于输出的缩减方式，取值为'none'、'mean'或'sum'。默认值：'mean'。

    输入：
        - **x** (Tensor) - 输入Tensor。shape可以是任意维。
        - **target** (Tensor) - 标签Tensor，其shape与 `x` 相同。

    输出：
        Tensor或Scalar，如果 `reduction` 为none'，则输出是Tensor，其形状与 `x` 相同。否则，它是Scalar。

    异常：
        - **TypeError** - `reduction` 不是str类型。
        - **TypeError** - `x` 或 `target` 都不是Tensor。
        - **TypeError** - `x` 或 `target` 的数据类型不支持。
