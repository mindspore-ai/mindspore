mindspore.mint.nn.functional.pad
================================
.. py:function:: mindspore.mint.nn.functional.pad(input, pad, mode='constant', value=None)

    根据参数 `pad` 对输入进行填充。

    .. warning::
        ``'circular'`` 模式暂不支持 Ascend。

    参数：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`(N, *)`， :math:`*` 代表任意附加维度。
        - **pad** (Union[tuple[int], list[int], Tensor]) - pad的填充位置。
          :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` 维度的 `input` 将会被填充。可根据以下示例以此类推：

          - 示例：若只需要填充输入tensor的最后一个维度，则 `pad` 的填充方式为 :math:`(\text{padding_left}, \text{padding_right})`;
          - 示例：若只需要填充输入tensor的最后两个维度，则 `pad` 的填充方式为 :math:`(\text{padding_left}, \text{padding_right}, \text{padding_top}, \text{padding_bottom})`;
          - 示例：若只需要填充输入tensor的最后三个维度，则 `pad` 的填充方式为 :math:`(\text{padding_left}, \text{padding_right}, \text{padding_top}, \text{padding_bottom}, \text{padding_front}, \text{padding_back})`;

        - **mode** (str，可选) - pad的填充模式，可选择 ``'constant'`` 、 ``'reflect'`` 、 ``'replicate'`` 或者 ``'circular'`` 。默认值： ``'constant'`` 。

          - 对于 ``'constant'`` 模式，请参考 :class:`mindspore.nn.ConstantPad1d` 作为示例来理解这个填充模式，并将这个模式扩展到n维。
          - 对于 ``'reflect'`` 模式，请参考 :class:`mindspore.nn.ReflectionPad1d` 作为示例来理解这个填充模式，reflect模式用于填充四维或五维输入的最后三个维度、三维或四维输入的最后两个维度，或者二维或三维输入的最后一个维度。
          - 对于 ``'replicate'`` 模式，请参考 :class:`mindspore.nn.ReplicationPad1d` 作为示例来理解这个填充模式，replicate模式用于填充四维或五维输入的最后三个维度、三维或四维输入的最后两个维度，或者二维或三维输入的最后一个维度。
          - 对于 ``'circular'`` 模式，circular模式用于将图像的像素从一侧循环地填充到另一侧。例如，右侧的像素将被替换为左侧的像素，底部的像素将被替换为顶部的像素。circular模式用于填充四维或五维输入的最后三个维度、三维或四维输入的最后两个维度，或者二维或三维输入的最后一个维度。当前该模式已在 CPU 后端支持上述场景；Ascend 后端不支持该模式。

        - **value** (Union[int, float, None]，可选) - 仅在 ``'constant'`` 模式下生效，设置在 ``'constant'`` 模式下的填充值，如果值为 ``None`` ，则会使用0作为默认填充值。默认值： ``None`` 。

    .. note::
        在 Ascend 后端，当前支持以下场景：

        - ``'constant'`` 模式：支持对输入最后 :math:`\frac{\text{len(pad)}}{2}` 个维度进行常量填充。 `pad` 的长度必须为偶数，且不能超过 `2 * input.ndim` 。
        - ``'reflect'`` 模式：仅支持1D/2D/3D反射填充。 `pad` 长度为2时，输入维度必须为2或3； `pad` 长度为4时，输入维度必须为3或4； `pad` 长度为6时，输入维度必须为4或5。
        - ``'replicate'`` 模式：仅支持1D/2D/3D复制填充。 `pad` 长度为2时，输入维度必须为2或3； `pad` 长度为4时，输入维度必须为3或4； `pad` 长度为6时，输入维度必须为4或5。
        - ``'circular'`` 模式：Ascend 后端不支持。

    返回：
        填充后的Tensor。

    异常：
        - **TypeError** - `pad` 不是全为int的tuple或者list。
        - **ValueError** - `pad` 的长度不为偶数。
        - **ValueError** - `mode` 为 ``'constant'`` 且 `pad` 的长度大于 `2 * input.ndim` 。
        - **ValueError** - `mode` 为 ``'reflect'`` 或 ``'replicate'`` 且 `pad` 的长度不为2、4或6。
        - **ValueError** - `mode` 为 ``'reflect'`` 或 ``'replicate'`` 且 `pad` 与 `input` 维度的组合不在 Ascend 支持场景中。
        - **ValueError** - `mode` 不为 ``'constant'`` 并且 `value` 既不是 ``None`` 也不是0。
        - **RuntimeError** - `mode` 为 ``'circular'`` 且运行在 Ascend 后端。
