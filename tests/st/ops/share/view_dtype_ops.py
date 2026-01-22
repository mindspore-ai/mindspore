# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Utility helpers for operation testing.

This module provides:
- ViewDtypeOpsFactory: constructs view dtype operator testcases and enables
  forward/gradient comparisons across backends.
"""
import pytest
import hashlib
import mindspore as ms
import numpy as np
from mindspore.ops.auto_generate.gen_ops_def import as_strided
from mindspore.graph.api import _pynative_executor
from tests.st.ops.share._internal.meta import OpsFactory
from tests.st.ops.share._internal.utils import make_tensor, OpSampleInput, OpErrorInput, ms_asnumpy, torch_asnumpy
from tests.st.ops.share._op_info.op_info import OpInfo


class ViewDtypeOpsFactory(OpsFactory):
    """Factory for view dtype ops testcases.

    Extends the common factory with net class tweaking and view dtype-specific
    sample input builders.
    """

    def test_op_reference(
            self,
            *,
            grad_cmp: bool = False,
    ):
        """Run reference parity tests against benchmark for all supported dtypes.

        Args:
            grad_cmp (bool): When True, restrict to floating dtypes and compare first-order gradients.
        """
        def get_md5(tensor):
            arr = ms_asnumpy(tensor) if isinstance(tensor, ms.Tensor) else torch_asnumpy(tensor)
            arr_bytes = np.ascontiguousarray(arr).tobytes()
            return hashlib.md5(arr_bytes).hexdigest()

        def sample_inputs_view_dtype_func(op_info: OpInfo, dtype, new_dtype, device):
            low = -5
            high = 5

            shapes = (
                (4, 4, 64),
                (0, 5, 64),
                (),
            )
            for shape in shapes:
                # normal tensor
                yield OpSampleInput(
                    op_input=make_tensor(shape, dtype=dtype, device=device, low=low, high=high),
                    op_args=(new_dtype,),
                    op_kwargs={},
                    sample_name=op_info.name + f"_normal_tensor_shape{shape}_dtype({dtype})_newdtype({new_dtype})",
                )

            # view tensor
            yield OpSampleInput(
                op_input=make_tensor((4, 4, 64), dtype=dtype, device=device, low=low, high=high).permute(1, 0, 2),
                op_args=(new_dtype,),
                op_kwargs={},
                sample_name=op_info.name + f"_view_tensor_shape(4, 4, 64)_permute_dtype({dtype})_newdtype({new_dtype})",
            )
            yield OpSampleInput(
                op_input=make_tensor((4, 64, 4), dtype=dtype, device=device, low=low, high=high).permute(2, 0, 1),
                op_args=(new_dtype,),
                op_kwargs={},
                sample_name=op_info.name + f"_view_tensor_shape(4, 64, 4)_permute_dtype({dtype})_newdtype({new_dtype})",
            )
            yield OpSampleInput(
                op_input=make_tensor((1, 5, 1), dtype=dtype, device=device, low=low, high=high).expand(5, 5, 64),
                op_args=(new_dtype,),
                op_kwargs={},
                sample_name=op_info.name + f"_view_tensor_shape(1, 5, 1)_expand_dtype({dtype})_newdtype({new_dtype})",
            )
            yield OpSampleInput(
                op_input=make_tensor((2, 5, 256), dtype=dtype, device=device, low=low, high=high)[1::2, 1:, ::2],
                op_args=(new_dtype,),
                op_kwargs={},
                sample_name=op_info.name + f"_view_tensor_shape(2, 5, 256)_slice_dtype({dtype})_newdtype({new_dtype})",
            )

        # pylint: disable=R1702
        try:
            if grad_cmp:
                self.supported_dtypes = tuple(d for d in self.supported_dtypes if d.is_floating_point or d.is_complex)
            for dtype in self.supported_dtypes:
                for new_dtype in self.supported_dtypes:
                    print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, "
                          f"dtype:{dtype}, new_dtype:{new_dtype} test_op_reference...")
                    is_same_element_size = dtype.itemsize == new_dtype.itemsize
                    for sample_input in sample_inputs_view_dtype_func(
                            self.op_info,
                            dtype,
                            new_dtype,
                            self._device,
                    ):
                        # scalar tensor will raise ValueError if elementsize is not same
                        if not is_same_element_size and sample_input.op_input.shape == ():
                            with pytest.raises(ValueError):
                                _ = self.op(sample_input.op_input, *sample_input.op_args, **sample_input.op_kwargs)
                                _pynative_executor.sync()
                        # strides[-1] must be 1 if elementsize is not same
                        elif not is_same_element_size and sample_input.op_input.stride(-1) != 1:
                            with pytest.raises(ValueError):
                                _ = self.op(sample_input.op_input, *sample_input.op_args, **sample_input.op_kwargs)
                                _pynative_executor.sync()
                        else:
                            if grad_cmp:
                                self.update_inputs([sample_input])
                                _ = self.grad_mindspore_impl()
                                _pynative_executor.sync()
                            else:
                                if self._device == 'ascend' and new_dtype == ms.bfloat16:
                                    # float() will use aclnnCast to cast bfloat16 to float32, bits of value will be
                                    # changed in some cases:
                                    # >> import mindspore as ms
                                    # >> import numpy as np
                                    # >> import torch
                                    # >> import torch_npu
                                    # >> x = ms.frombuffer(b'\xff\xff', dtype=ms.bfloat16)
                                    # >> x_bytes = x.float().asnumpy().tobytes()
                                    # >> print(x_bytes)
                                    # >> b'\xff\xff\xff\x7f'
                                    # >> y = torch.frombuffer(b'\xff\xff', dtype=torch.bfloat16)
                                    # >> y_bytes = y.float().numpy().tobytes()
                                    # >> print(y_bytes)
                                    # >> b'\x00\x00\xff\xff'
                                    # >> y = y.npu()
                                    # >> y_bytes = y.float().cpu().numpy().tobytes()
                                    # >> print(y_bytes)
                                    # >> b'\xff\xff\xff\x7f'
                                    self.compare_with_torch(sample_inputs=sample_input)
                                else:
                                    self.update_inputs([sample_input])
                                    ms_out = self.forward_mindspore_impl()
                                    pt_out = self.forward_pytorch_impl()
                                    for ms_outi, pt_outi in zip(ms_out, pt_out):
                                        assert get_md5(ms_outi) == get_md5(pt_outi)
        except Exception as e:
            print(f"\ntest_op_reference failed:"
                  f"\nop_name: {self.op_name}"
                  f"\nmode: {self._context_mode}"
                  f"\n{sample_input.summary(True)}")
            raise e

    def test_op_error(self):
        """
        Test error cases.
        """
        def error_inputs_view_dtype_func(op_info, dtypes, device):
            for dtype in dtypes:
                for new_dtype in dtypes:
                    old_element_size = dtype.itemsize
                    new_element_size = new_dtype.itemsize

                    if new_element_size <= old_element_size:
                        continue

                    size_ratio = new_element_size // old_element_size
                    low = -5
                    high = 5
                    error_tensor = make_tensor((8, 8, size_ratio + 1), low=low, high=high, dtype=dtype, device=device)

                    yield OpErrorInput(
                        op_sample_input=OpSampleInput(
                            op_input=error_tensor,
                            op_args=(new_dtype,),
                            op_kwargs={},
                            sample_name=op_info.name,
                        ),
                        op_error_type=ValueError,
                        op_error_info=f'last dimension must be divisible by size_ratio {size_ratio}',
                    )

                    yield OpErrorInput(
                        op_sample_input=OpSampleInput(
                            op_input=error_tensor[:, :, 1:],
                            op_args=(new_dtype,),
                            op_kwargs={},
                            sample_name=op_info.name,
                        ),
                        op_error_type=ValueError,
                        op_error_info=f'storage offset must be divisible by size_ratio {size_ratio}',
                    )

                    tensor = make_tensor((8, 8, size_ratio), low=low, high=high, dtype=dtype, device=device)
                    error_tensor = as_strided(tensor, (8, 8, size_ratio), (size_ratio, 1, 1))

                    yield OpErrorInput(
                        op_sample_input=OpSampleInput(
                            op_input=error_tensor,
                            op_args=(new_dtype,),
                            op_kwargs={},
                            sample_name=op_info.name,
                        ),
                        op_error_type=ValueError,
                        op_error_info=f'old_strides(1) must be divisible by size_ratio {size_ratio}',
                    )

        try:
            print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, test_view_dtype_error...")
            for error_input in error_inputs_view_dtype_func(self.op_info, self.supported_dtypes, device=self._device):
                _sample_input = error_input.op_sample_input
                _error_type = error_input.op_error_type

                with pytest.raises(_error_type):
                    self.op(_sample_input.op_input, *_sample_input.op_args, **_sample_input.op_kwargs)
                    _pynative_executor.sync()
        except Exception as e:
            print(f"\ntest_view_dtype_error catch expect {_error_type.__name__} failed, but got {type(e).__name__}:"
                  f"\nop_name: {self.op_name}"
                  f"\nmode: {self._context_mode}"
                  f"\nsample_input: {_sample_input.summary()}"
                  f"\nerror_info: {error_input.op_error_info}")
            raise e
