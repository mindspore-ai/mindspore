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

#ifndef MINDSPORE_CCSRC_PYBIND_API_TENSOR_BUFFER_H_
#define MINDSPORE_CCSRC_PYBIND_API_TENSOR_BUFFER_H_

#include <pybind11/pybind11.h>
#include "ir/tensor.h"

namespace mindspore {
namespace tensor {
namespace py = pybind11;

/**
 * @brief Copy a CPU Tensor into a writable Python buffer.
 *
 * This function backs the internal ``_save_tensor_to_buf`` Python API used by checkpoint code. It waits for
 * pending forward work, creates a temporary Tensor view over ``buf``, and copies the source Tensor into that view.
 * The pybind binding intentionally keeps the Python GIL while Python objects and buffer memory are accessed. The
 * pipeline wait may temporarily release the GIL, but it is reacquired before the buffer view and host copy are used.
 *
 * @param tensor Source MindSpore Tensor. Its data must be materialized on CPU-compatible storage.
 * @param buf Writable Python object implementing the buffer protocol, such as ``bytearray`` or ``memoryview``.
 * @param dtype Data type used to interpret the destination buffer. Checkpoint callers should pass ``tensor.dtype``.
 * @param shape Source shape metadata retained for symmetry with the load API. It is not validated or used to copy.
 * @param count Number of destination elements. ``-1`` means all complete elements after ``offset``.
 * @param offset Byte offset from the beginning of ``buf``.
 *
 * @note For a lossless checkpoint copy, ``dtype``, ``shape``, and ``count`` must describe the source Tensor exactly.
 * @note The GIL serializes the critical buffer-access section between Python threads. Callers must still order
 * dependent accesses to the same buffer region and synchronize writers that do not use the Python GIL.
 */
void SaveTensorToBuf(const py::object &tensor, const py::object &buf, const py::object &dtype,
                     const std::vector<int64_t> &shape, int64_t count = -1, int64_t offset = 0);

/**
 * @brief Create a shaped Tensor that aliases a Python buffer without copying its data.
 *
 * This function backs the internal ``_load_tensor_from_buf`` Python API used by checkpoint code. It first creates a
 * flat Tensor view over ``buf`` and then creates a Tensor with ``shape`` that reuses the same device address. The
 * Python GIL remains held while the buffer is inspected, its lifetime is retained, and the Tensor objects are built.
 *
 * @param buf Python object implementing the buffer protocol. The returned Tensor keeps this object alive.
 * @param dtype Data type used to interpret bytes in ``buf``.
 * @param shape Logical shape of the returned Tensor.
 * @param count Number of elements exposed from ``buf``. ``-1`` means all complete elements after ``offset``.
 * @param offset Byte offset from the beginning of ``buf``.
 * @return A Python MindSpore Tensor that shares storage with ``buf``.
 *
 * @note The product of ``shape`` must equal ``count`` (or the inferred count); this relation is not validated here.
 * @note Because the result is zero-copy, changes to the buffer are visible through the Tensor and vice versa. Callers
 * must coordinate later accesses to shared or overlapping buffer regions after this function returns.
 */
py::object LoadTensorFromBuf(const py::object &buf, const py::object &dtype, const std::vector<int64_t> &shape,
                             int64_t count = -1, int64_t offset = 0);

}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_TENSOR_BUFFER_H_
