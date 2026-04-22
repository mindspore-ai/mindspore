mindspore.ops.move_to
======================

.. py:function:: mindspore.ops.move_to(input, to="CPU", blocking=True)

    拷贝tensor到目标设备，包含同步和异步两种方式，默认是同步方式。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        该接口当前仅支持graph mode，并且jit_level为O0或O1。

    参数：
        - **input** (Union[Tensor, list[int], tuple[int]]) - 输入tensor，当输入为list和tuple时会先转换为tensor再进行拷贝。
        - **to** (str，可选) - 指定目标设备，可选值为 ``"Ascend"``， ``"CPU"`` 。默认 ``"CPU"`` 。
        - **blocking** (bool，可选) - 是否使用同步拷贝。默认 ``True`` 表示同步拷贝。

    返回：
        目标设备上的新tensor。
