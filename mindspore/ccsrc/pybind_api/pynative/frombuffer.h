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

#ifndef MINDSPORE_CCSRC_TENSOR_FROMBUFFER_H_
#define MINDSPORE_CCSRC_TENSOR_FROMBUFFER_H_

#include <pybind11/pybind11.h>
#include "ir/tensor.h"

namespace mindspore {
namespace tensor {
namespace py = pybind11;
/**
 * @brief Create a Tensor from a Python buffer object (zero-copy)
 *
 * This function creates a Tensor that shares memory with the provided Python buffer object,
 * avoiding data copying for better performance.
 *
 * @param buffer Python object supporting buffer protocol (bytes, bytearray, numpy array, etc.)
 * @param dtype Data type of the Tensor to create
 * @param count Number of elements to create (-1 for automatic calculation)
 * @param offset Byte offset from the start of the buffer
 * @return A Tensor object sharing memory with the buffer
 *
 * @note The created Tensor shares memory with the Python buffer. Modifying the buffer
 *       while the Tensor is in use may lead to undefined behavior.
 *
 * @throws Exception if:
 *   - Buffer protocol is not supported by the object
 *   - Buffer retrieval fails
 *   - Parameters are invalid (negative offset, insufficient data, etc.)
 */

py::object TensorFrombuffer(const py::object &buffer, const py::object &dtype, int64_t count = -1, int64_t offset = 0);

}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TENSOR_FROMBUFFER_H_
