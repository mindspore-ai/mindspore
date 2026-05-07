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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_TENSOR_MEMORY_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_TENSOR_MEMORY_H_

#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "utils/current_state.h"

namespace mindspore {
namespace pyboost {
using DeviceContext = device::DeviceContext;

PYBOOST_API void CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                                          const tensor::TensorPtr &tensor);
PYBOOST_API void CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                                          const std::optional<tensor::TensorPtr> &val);
PYBOOST_API void CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                                          const ValueTuplePtr &value_tuple);
template <typename T>
void CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                              const std::vector<T> &inputs) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    CreateInputTensorAddress(device_context, stream_id, index, inputs[i]);
  }
}

PYBOOST_API void CreateOutputTensorAddress(const DeviceContext *device_context, size_t stream_id,
                                           const std::vector<tensor::TensorPtr> &outputs);
PYBOOST_API void CreateOutputTensorAddress(const DeviceContext *device_context, size_t stream_id,
                                           const tensor::TensorPtr &output_tensor, size_t size);

PYBOOST_API void CheckMemoryOverlap(const std::vector<tensor::TensorPtr> &inputs,
                                    const std::vector<tensor::TensorPtr> &outputs);
}  // namespace pyboost

namespace kernel {
namespace pyboost {
// Put input tensors on device (allocate device memory and copy data to device)
template <typename... Args>
void TensorToDevice(const ::mindspore::kernel::pyboost::OpPtr &op, const Args &...args) {
  auto device_context = op->device_context();
  auto stream_id = op->stream_id();
  size_t index = 0;
  auto add_index = [&index]() { return index++; };
  (::mindspore::pyboost::CreateInputTensorAddress(device_context, stream_id, add_index(), args), ...);
}

// Allocate device memory for output tensors
inline void TensorMalloc(const ::mindspore::kernel::pyboost::OpPtr &op, const std::vector<tensor::TensorPtr> &outputs) {
  auto device_context = op->device_context();
  auto stream_id = op->stream_id();
  ::mindspore::pyboost::CreateOutputTensorAddress(device_context, stream_id, outputs);
}

// Overloads using CurrentDeviceContext/CurrentStream (no OpRunner dependency)
template <typename... Args>
void TensorToDevice(const Args &...args) {
  auto device_context = CurrentDeviceContext::get();
  auto stream_id = CurrentStream::id();
  size_t index = 0;
  auto add_index = [&index]() { return index++; };
  (::mindspore::pyboost::CreateInputTensorAddress(device_context, stream_id, add_index(), args), ...);
}

inline void TensorMalloc(const std::vector<tensor::TensorPtr> &outputs) {
  auto device_context = CurrentDeviceContext::get();
  auto stream_id = CurrentStream::id();
  ::mindspore::pyboost::CreateOutputTensorAddress(device_context, stream_id, outputs);
}

inline void TensorMalloc(const tensor::TensorPtr &output) { TensorMalloc(std::vector<tensor::TensorPtr>{output}); }

template <typename... Ts>
void TensorMalloc(const std::tuple<Ts...> &outputs) {
  std::vector<tensor::TensorPtr> vec;
  std::apply([&vec](const auto &...ts) { (vec.push_back(ts), ...); }, outputs);
  TensorMalloc(vec);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_TENSOR_MEMORY_H_
