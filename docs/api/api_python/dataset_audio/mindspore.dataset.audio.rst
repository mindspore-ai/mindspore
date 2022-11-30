此模块用于音频数据增强，包括 `transforms` 和 `utils` 两个子模块。
`transforms` 是一个高性能音频数据增强模块，支持常见的音频数据增强操作。
`utils` 提供了一些音频处理的工具方法。

API样例中常用的导入模块如下：

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.audio as audio

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- AudioTensorOperation，所有音频数据处理操作的基类，派生自TensorOperation。