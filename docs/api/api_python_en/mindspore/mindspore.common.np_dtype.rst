mindspore.common.np_dtype
===========================

.. class:: mindspore.common.np_dtype

    NumPy data type for MindSpore.

    The actual path of ``np_dtype`` is ``/mindspore/common/np_dtype.py``.
    Run the following command to import the package:

    .. code-block::

        from mindspore.common import np_dtype

    * **Numeric Type**

      ============================   =================
      Type                            Description
      ============================   =================
      ``bfloat16``                   The ``bfloat16`` data type under NumPy. This type is only used to construct Tensor of type ``bfloat16``, and does not guarantee the full computing power under NumPy. It takes effect only if the version of NumPy at runtime is not less than the version of NumPy at compilation, and the major versions are the same.
      ============================   =================
