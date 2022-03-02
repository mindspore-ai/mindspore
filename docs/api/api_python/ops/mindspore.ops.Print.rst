mindspore.ops.Print
===================

.. py:class:: mindspore.ops.Print

    将输入Tensor或string进行打印输出。
	
    默认打印在屏幕上。也可以保存在文件中，通过 `context` 设置 `print_file_path` 参数。一旦设置，输出将保存在指定文件中。通过函数 :func:`mindspore.parse_print` 可以重新加载数据。获取更多信息，请查看 :func:`mindspore.context.set_context` 和 :func:`mindspore.parse_print` 。

    .. note::
        在PyNative模式下，请使用Python print函数。在Graph模式下，bool、int和float将被转换为Tensor进行打印，str保持不变。

    **输入：**

    - **input_x** (Union[Tensor, bool, int, float, str]) - Print的输入。支持多个输入，用'，'分隔。

    **输出：**

    Tensor，数据类型和shape与 `input_x` 相同。

    **异常：**

    - **TypeError** - `input_x` 不是Tensor、bool、int、float或str。