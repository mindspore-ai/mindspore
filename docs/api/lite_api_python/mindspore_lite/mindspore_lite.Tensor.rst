mindspore_lite.Tensor
=====================

.. py:class:: mindspore_lite.Tensor(tensor=None)

    张量类，在Mindsporlite中定义了一个张量。

    参数：
        - **tensor** (Tensor，可选) - 被存储在新张量中的数据，可以是其它Tensor。默认值：None。

    异常：
        - **TypeError** - `tensor` 不是Tensor类型或None。

    .. py:method:: get_data_size()

        获取张量的数据大小，即 :math:`data\_size = element\_num * data\_type` 。

        返回：
            int，张量数据的数据大小。

    .. py:method:: get_data_to_numpy()

        从张量获取numpy对象的数据。

        返回：
            numpy.ndarray，张量数据中的numpy对象。

    .. py:method:: get_data_type()

        获取张量的数据类型。

        返回：
            DataType，张量的数据类型。

    .. py:method:: get_element_num()

        获取张量的元素数。

        返回：
            int，张量数据的元素数。

    .. py:method:: get_format()

        获取张量的格式。

        返回：
            Format，张量的格式。

    .. py:method:: get_shape()

        获取张量的形状。

        返回：
        list[int]，张量的形状。

    .. py:method:: get_tensor_name()

        获取张量的名称。

        返回：
            str，张量的名称。

    .. py:method:: set_data_from_numpy(numpy_obj)

        从numpy对象设置张量的数据。

        参数：
            - **numpy_obj** (numpy.ndarray) - numpy对象。

        异常：
            - **TypeError** - `numpy_obj` 不是numpy.ndarray类型。
            - **RuntimeError** - `numpy_obj` 的数据类型与张量的数据类型不等价。
            - **RuntimeError** - `numpy_obj` 的数据大小与张量的数据大小不相等。

    .. py:method:: set_data_type(data_type)

        设置张量的数据类型。

        参数：
            - **data_type** (DataType) - 张量的数据类型。

        异常：
            - **TypeError** - `data_type` 不是DataType类型。

    .. py:method:: set_format(tensor_format)

        设置张量的格式。

        参数：
            - **tensor_format** (Format) - 张量的格式。

        异常：
            - **TypeError** - `tensor_format` 不是Format类型。

    .. py:method:: set_shape(shape)

        设置张量的形状。

        参数：
            - **shape** (list[int]) - 张量的形状。

        异常：
            - **TypeError** - `shape` 不是list类型。
            - **TypeError** - `shape` 是list类型，但元素不是int类型。

    .. py:method:: set_tensor_name(tensor_name)

        设置张量的名称。

        参数：
            - **tensor_name** (str) - 张量的名称。

        异常：
            - **TypeError** - `tensor_name` 不是str类型。
