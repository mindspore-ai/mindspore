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
#include "include/runtime/pipeline/pipeline.h"
#include "include/pynative/utils/pynative_utils.h"
#include "include/pynative/forward/pyboost/fallback.h"

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
  mindspore::pynative::RegPyNativeExecutor(m);
  mindspore::pynative::RegisterPyBoostFunction(m);
  mindspore::pynative::RegisterCustomizeFunction(m);
  mindspore::pynative::RegisterCellBackwardHookFunction(m);
  mindspore::pynative::RegisterDetachFunction(m);
  mindspore::pynative::RegisterFunctional(m);
  mindspore::pynative::RegOpCall(m);
  mindspore::pynative::RegFallback(m);
  mindspore::pynative::distributed::RegReducer(m);
  mindspore::pynative::autograd::RegBackwardFunction(m);
  mindspore::pynative::autograd::RegBackwardNode(m);
  mindspore::pynative::autograd::RegFunctionBase(m);
  RegTensorDoc(m);
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
}

}  // namespace mindspore
