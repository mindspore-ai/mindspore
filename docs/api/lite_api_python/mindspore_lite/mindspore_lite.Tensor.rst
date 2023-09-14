mindspore_lite.Tensor
=====================

.. py:class:: mindspore_lite.Tensor(tensor=None, shape=None, dtype=None, device=None)

    `Tensor` 类，在Mindspore Lite中定义一个张量。

    参数：
        - **tensor** (Tensor，可选) - 被存储在新Tensor中的数据，数据可以是来自其它Tensor。默认值： ``None`` 。
        - **shape** (list，可选) - Tensor的shape信息。默认值： ``None`` 。
        - **dtype** (DataType，可选) - Tensor的dtype信息。默认值： ``None`` 。
        - **device** (str，可选) - Tensor的device信息。默认值： ``None`` 。

    异常：
        - **TypeError** - `tensor` 既不是Tensor类型也不是 ``None`` 。

    .. py:method:: data_size
        :property:

        获取Tensor的数据大小。

        Tensor的数据大小 = Tensor的元素数量 * Tensor的单位数据类型对应的size。

        返回：
            int，Tensor的数据大小。

    .. py:method:: device
        :property:

        获取Tensor的device信息。

        返回：
            str，Tensor的device信息。

    .. py:method:: dtype
        :property:

        获取Tensor的数据类型。

        返回：
            DataType，Tensor的数据类型。

    .. py:method:: element_num
        :property:

        获取Tensor的元素数。

        返回：
            int，Tensor数据的元素数。

    .. py:method:: format
        :property:

        获取Tensor的格式。

        返回：
            Format，Tensor的格式。

    .. py:method:: get_data_to_numpy()

        从Tensor获取数据传给numpy对象。

        返回：
            numpy.ndarray，Tensor数据中的numpy对象。

    .. py:method:: name
        :property:

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

    .. py:method:: shape
        :property:

        获取Tensor的shape。

        返回：
            list[int]，Tensor的shape。