mindspore_lite.Tensor
=====================

.. py:class:: mindspore_lite.Tensor(tensor=None)

    `Tensor` 类，在Mindspore Lite中定义一个张量。

    参数：
        - **tensor** (Tensor，可选) - 被存储在新Tensor中的数据，数据可以是来自其它Tensor。默认值：None。

    异常：
        - **TypeError** - `tensor` 既不是Tensor类型也不是None。

    .. py:method:: get_data_size()

        获取Tensor的数据大小。

        Tensor的数据大小 = Tensor的元素数量 * Tensor的单位数据类型对应的size。

        返回：
            int，Tensor的数据大小。

    .. py:method:: get_data_to_numpy()

        从Tensor获取数据传给numpy对象。

        返回：
            numpy.ndarray，Tensor数据中的numpy对象。

    .. py:method:: get_data_type()

        获取Tensor的数据类型。

        返回：
            DataType，Tensor的数据类型。

    .. py:method:: get_element_num()

        获取Tensor的元素数。

        返回：
            int，Tensor数据的元素数。

    .. py:method:: get_format()

        获取Tensor的格式。

        返回：
            Format，Tensor的格式。

    .. py:method:: get_shape()

        获取Tensor的shape。

        返回：
            list[int]，Tensor的shape。

    .. py:method:: get_tensor_name()

        获取Tensor的名称。

        返回：
            str，Tensor的名称。

    .. py:method:: set_data_from_numpy(numpy_obj)

        从numpy对象获取数据传给Tensor。

        参数：
            - **numpy_obj** (numpy.ndarray) - numpy对象。

        异常：
            - **TypeError** - `numpy_obj` 不是numpy.ndarray类型。
            - **RuntimeError** - `numpy_obj` 的数据类型与Tensor的数据类型不等价。
            - **RuntimeError** - `numpy_obj` 的数据大小与Tensor的数据大小不相等。

    .. py:method:: set_data_type(data_type)

        设置Tensor的数据类型。

        参数：
            - **data_type** (DataType) - Tensor的数据类型。有关详细信息，请参见 `DataType <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.DataType.html>`_ 。

        异常：
            - **TypeError** - `data_type` 不是DataType类型。

    .. py:method:: set_format(tensor_format)

        设置Tensor的格式。

        参数：
            - **tensor_format** (Format) - Tensor的格式。有关详细信息，请参见 `Format <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Format.html>`_ 。

        异常：
            - **TypeError** - `tensor_format` 不是Format类型。

    .. py:method:: set_shape(shape)

        设置Tensor的shape。

        参数：
            - **shape** (list[int]) - Tensor的shape。

        异常：
            - **TypeError** - `shape` 不是list类型。
            - **TypeError** - `shape` 是list类型，但元素不是int类型。

    .. py:method:: set_tensor_name(tensor_name)

        设置Tensor的名称。

        参数：
            - **tensor_name** (str) - Tensor的名称。

        异常：
            - **TypeError** - `tensor_name` 不是str类型。
