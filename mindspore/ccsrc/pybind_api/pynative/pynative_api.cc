/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "pybind_api/pynative/pynative_api.h"
#include "pybind11/pybind11.h"

#include "include/frontend/jit/ps/pipeline_interface.h"
#include "pybind_api/pynative/frombuffer.h"
#include "pybind_api/tensor_buffer.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/pynative/utils/pynative_utils.h"
#include "include/pynative/forward/pyboost/fallback.h"
#include "include/utils/pynative/dispatch/ms_dispatch.h"
#include "include/utils/pytree.h"
#include "include/utils/tensor_py.h"

namespace mindspore {
namespace py = pybind11;

void RegPyNativeModule(py::module *m) {
  RegStorage(m);
  mindspore::tensor::RegMetaTensor(m);
  mindspore::tensor::RegCSRTensor(m);
  mindspore::tensor::RegCOOTensor(m);
  mindspore::tensor::RegRowTensor(m);
  mindspore::tensor::RegMapTensor(m);
  mindspore::tensor::RegPyTensor(m);
  mindspore::tensor::RegTensorUtilsApi(m);
  mindspore::pynative::RegPyNativeExecutor(m);
  mindspore::pynative::RegisterPyBoostFunction(m);
  mindspore::pynative::RegisterFunctional(m);
  mindspore::pynative::RegOpCall(m);
  mindspore::pynative::RegFallback(m);
  mindspore::pynative::RegMsDispatch(m);
  mindspore::pynative::distributed::RegReducer(m);
  mindspore::pynative::autograd::RegBackwardFunction(m);
  mindspore::pynative::autograd::RegBackwardNode(m);
  mindspore::pynative::autograd::RegTensorMeta(m);
  mindspore::pynative::autograd::RegFunctionBase(m);
  mindspore::pytree::RegPyTree(m);
  RegTensorDoc(m);
  (void)m->def(
    "_unsafe_set_version_counter",
    [](const std::vector<tensor::TensorPtr> &tensors, const std::vector<int64_t> &versions) {
      auto tensors_len = tensors.size();
      auto versions_len = versions.size();
      if (tensors_len != versions_len) {
        MS_LOG(EXCEPTION) << "tensors size(" << tensors_len << ") != versions size(" << versions_len << ")";
      }
      runtime::Pipeline::Get().WaitFrontend();
      for (size_t i = 0; i < tensors_len; ++i) {
        MS_EXCEPTION_IF_NULL(tensors[i]);
        tensors[i]->SetVersionValue(static_cast<uint32_t>(versions[i]));
      }
    },
    "Unsafe set version counter for tensors.");
  (void)m->def("disable_multi_thread", &mindspore::runtime::Pipeline::DisableMultiThreadAfterFork,
               "Disable multi thread");
  (void)m->def("frombuffer", &mindspore::tensor::TensorFrombuffer, py::arg("buffer"), py::arg("dtype"),
               py::arg("count") = -1, py::arg("offset") = 0,
               R"(
                Create a 1-D tensor from an object implements the buffer interface.
                The returned tensor shares the same memory space as the buffer.

                Args:
                    buffer (buffer): An object that exposes the buffer interface.
                    dtype (mindspore.dtype): The desired data type for the tensor.
                    count (int, optional): The number of items to read. Default: ``-1`` .
                    offset (int, optional): The offset in bytes. Default: ``0`` .

                Returns:
                    Tensor, a 1-D tensor containing the data from the buffer.

                Raises:
                    RuntimeError: If `count` is a negative number less than -1.
                    RuntimeError: If the buffer length is less than or equal to 0, or if `count` is 0.
                    RuntimeError: If `offset` is not within the range [0, buffer length - 1].
                    RuntimeError: When `count` is -1, if (buffer length - `offset`) is
                                  not divisible by the element size.
                    RuntimeError: If the remaining buffer size after `offset` is insufficient to
                                  accommodate the requested number of elements.
               
                Examples:
                    >>> import mindspore
                    >>> data = bytearray([1, 2, 3, 4, 5, 6])
                    >>> tensor = mindspore.frombuffer(data, dtype=mindspore.uint8)
                    >>> print(tensor)
                    [1 2 3 4 5 6]
    )");
  (void)m->def("_save_tensor_to_buf", &mindspore::tensor::SaveTensorToBuf, py::arg("tensor"), py::arg("buf"),
               py::arg("dtype"), py::arg("shape"), py::arg("count") = -1, py::arg("offset") = 0,
               R"(
                Copy a CPU Tensor into a writable buffer for checkpoint save.

                ``dtype``, ``count``, and ``offset`` describe the destination; ``shape`` is metadata only. The GIL is
                held during buffer access and copying. Pipeline waiting may release it temporarily; it is reacquired
                before the buffer is touched. Omitting ``count`` and ``offset`` uses the whole buffer from byte 0.

                Examples:
                    >>> import mindspore
                    >>> tensor = mindspore.Tensor([1, 2, 3], mindspore.int32)
                    >>> buf = bytearray(12)
                    >>> mindspore._save_tensor_to_buf(tensor, buf, mindspore.int32, (3,))
    )");
  (void)m->def("_load_tensor_from_buf", &mindspore::tensor::LoadTensorFromBuf, py::arg("buf"), py::arg("dtype"),
               py::arg("shape"), py::arg("count") = -1, py::arg("offset") = 0,
               R"(
                Create a shaped Tensor that shares memory with a buffer for checkpoint load.

                ``dtype``, ``count``, and ``offset`` describe the source; ``shape`` must match the element count. The GIL
                is held while inspecting the buffer and constructing the Tensor, which also keeps the buffer alive.
                Omitting ``count`` and ``offset`` uses the whole buffer from byte 0.

                Examples:
                    >>> import mindspore
                    >>> buf = bytearray(12)
                    >>> tensor = mindspore._load_tensor_from_buf(buf, mindspore.int32, (3,))
    )");
}

}  // namespace mindspore
