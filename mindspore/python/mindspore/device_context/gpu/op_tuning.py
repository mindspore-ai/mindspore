# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Op tuning interfaces."""

try:
    from mindspore._c_expression import GPUOpTuningConf
except ImportError:
    pass
from .device import _is_supported

function_status = {'conv_fprop_algo': False, 'conv_wgrad_algo': False, 'conv_dgrad_algo': False}


def conv_fprop_algo(mode):
    """
    Specifies convolution forward algorithm.
    For detailed information, please refer to `NVIDIA cuDNN about cudnnConvolutionForward
    <https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html>`_.

    Args:
        mode (str): Convolution forward algorithm. If not configured, the framework defaults to ``'normal'``.
          The value range is as follows:

          - normal: Use the cuDNN's heuristic search algorithm, the appropriate convolution algorithm will be quickly
            selected based on the convolution shape and type. This parameter does not guarantee optimal performance.
          - performance: Use the cuDNN's trial search algorithm, all convolution algorithms will be trial run based on
            the convolution shape and type, and the optimal algorithm will be selected. This parameter ensures optimal
            performance.
          - implicit_gemm: This algorithm expresses the convolution as a matrix product without actually explicitly
            forming the matrix that holds the input tensor data.
          - precomp_gemm: This algorithm expresses convolution as a matrix product without actually explicitly
            forming the matrix that holds the input tensor data, but still needs some memory workspace to precompute
            some indices in order to facilitate the implicit construction of the matrix that holds the input tensor
            data.
          - gemm: This algorithm expresses the convolution as an explicit matrix product. A significant memory
            workspace is needed to store the matrix that holds the input tensor data.
          - direct: This algorithm expresses the convolution as a direct convolution, without implicitly or explicitly
            doing a matrix multiplication.
          - fft: This algorithm uses the Fast-Fourier Transform approach to compute the convolution. A significant
            memory workspace is needed to store intermediate results.
          - fft_tiling: This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles. A
            significant memory workspace is needed to store intermediate results but less than fft algorithm for large
            size images.
          - winograd: This algorithm uses the Winograd Transform approach to compute the convolution. A reasonably
            sized workspace is needed to store intermediate results.
          - winograd_nonfused: This algorithm uses the Winograd Transform approach to compute the convolution. A
            significant workspace may be needed to store intermediate results.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.gpu.op_tuning.conv_fprop_algo("performance")
    """
    if not function_status['conv_fprop_algo']:
        function_status['conv_fprop_algo'] = True
        if not _is_supported():
            return
    conv_fprop_algo_mode = ["normal", "performance", "implicit_gemm", "precomp_gemm", "gemm", "direct",
                            "fft", "fft_tiling", "winograd", "winograd_nonfused"]
    if mode in conv_fprop_algo_mode:
        GPUOpTuningConf.get_instance().set_conv_fprop_algo(mode)
    else:
        raise ValueError(
            f"For 'mindspore.device_context.gpu.op_tuning.conv_fprop_algo', the argument must be in "
            f"{conv_fprop_algo_mode} but got {mode}."
        )


def conv_wgrad_algo(mode):
    """
    Specifies convolution filter grad algorithm.
    For detailed information, please refer to `NVIDIA cuDNN
    <https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html>`_.

    Args:
        mode (str): Convolution filter grad algorithm. If not configured, the framework defaults to ``'normal'``.
          The value range is as follows:

          - normal: Use the cuDNN's heuristic search algorithm, the appropriate convolution algorithm will be quickly
            selected based on the convolution shape and type. This parameter does not guarantee optimal performance.
          - performance: Use the cuDNN's trial search algorithm, all convolution algorithms will be trial run based on
            the convolution shape and type, and the optimal algorithm will be selected. This parameter ensures optimal
            performance.
          - algo_0: This algorithm expresses the convolution as a sum of matrix products without actually explicitly
            forming the matrix that holds the input tensor data. The sum is done using the atomic add operation, thus
            the results are non-deterministic.
          - algo_1: This algorithm expresses the convolution as a matrix product without actually explicitly forming
            the matrix that holds the input tensor data. The results are deterministic.
          - algo_3: This algorithm is similar to algo_0 but uses some small workspace to precompute some indices. The
            results are also non-deterministic.
          - fft: This algorithm uses a Fast-Fourier Transform approach to compute the convolution. A significant memory
            workspace is needed to store intermediate results. The results are deterministic.
          - fft_tiling: This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles. A
            significant memory workspace is needed to store intermediate results but less than fft for large size
            images. The results are deterministic.
          - winograd_nonfused: This algorithm uses the Winograd Transform approach to compute the convolution. A
            significant workspace may be needed to store intermediate results. The results are deterministic.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.gpu.op_tuning.conv_wgrad_algo("performance")
    """
    if not function_status['conv_wgrad_algo']:
        function_status['conv_wgrad_algo'] = True
        if not _is_supported():
            return
    conv_wgrad_algo_mode = ["normal", "performance", "algo_0", "algo_1", "fft", "algo_3",
                            "fft_tiling", "winograd_nonfused"]

    if mode in conv_wgrad_algo_mode:
        GPUOpTuningConf.get_instance().set_conv_wgrad_algo(mode)
    else:
        raise ValueError(
            f"For 'mindspore.device_context.gpu.op_tuning.conv_wgrad_algo', the argument must be in "
            f"{conv_wgrad_algo_mode} but got {mode}."
        )


def conv_dgrad_algo(mode):
    """
    Specifies convolution data grad algorithm.
    For detailed information, please refer to `NVIDIA cuDNN
    <https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html>`_.

    Args:
        mode (str): Convolution data grad algorithm. If not configured, the framework defaults to ``'normal'``.
          The value range is as follows:

          - normal: Use the cuDNN's heuristic search algorithm, the appropriate convolution algorithm will be quickly
            selected based on the convolution shape and type. This parameter does not guarantee optimal performance.
          - performance: Use the cuDNN's trial search algorithm, all convolution algorithms will be trial run based on
            the convolution shape and type, and the optimal algorithm will be selected. This parameter ensures optimal
            performance.
          - algo_0: This algorithm expresses the convolution as a sum of matrix products without actually explicitly
            forming the matrix that holds the input tensor data. The sum is done using the atomic add operation, thus
            the results are non-deterministic.
          - algo_1: This algorithm expresses the convolution as a matrix product without actually explicitly forming
            the matrix that holds the input tensor data. The results are deterministic.
          - fft: This algorithm uses a Fast-Fourier Transform approach to compute the convolution. A significant memory
            workspace is needed to store intermediate results. The results are deterministic.
          - fft_tiling: This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles. A
            significant memory workspace is needed to store intermediate results but less than fft for large size
            images. The results are deterministic.
          - winograd: This algorithm uses the Winograd Transform approach to compute the convolution. A reasonably
            sized workspace is needed to store intermediate results. The results are deterministic.
          - winograd_nonfused: This algorithm uses the Winograd Transform approach to compute the convolution. A
            significant workspace may be needed to store intermediate results. The results are deterministic.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.gpu.op_tuning.conv_dgrad_algo("performance")
    """
    if not function_status['conv_dgrad_algo']:
        function_status['conv_dgrad_algo'] = True
        if not _is_supported():
            return
    conv_dgrad_algo_mode = ["normal", "performance", "algo_0", "algo_1", "fft", "fft_tiling",
                            "winograd", "winograd_nonfused"]

    if mode in conv_dgrad_algo_mode:
        GPUOpTuningConf.get_instance().set_conv_dgrad_algo(mode)
    else:
        raise ValueError(
            f"For 'mindspore.device_context.gpu.op_tuning.conv_dgrad_algo', the argument must be in "
            f"{conv_dgrad_algo_mode} but got {mode}."
        )
