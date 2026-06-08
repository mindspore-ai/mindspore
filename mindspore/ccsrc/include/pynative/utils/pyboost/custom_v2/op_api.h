/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_EXTENSION_CUSTOM_OP_API_H_
#define MINDSPORE_CCSRC_EXTENSION_CUSTOM_OP_API_H_

#include <vector>
#include <optional>
#include <memory>
#include <type_traits>
#include <functional>

#include "include/pynative/utils/pyboost/custom/tensor.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "include/pynative/utils/runtime/task/device_task.h"
#include "include/pynative/utils/runtime/op_runner.h"
#include "utils/current_state.h"
#include "utils/device_manager_conf.h"

namespace ms {

/// @brief Get device context from DeviceManagerConf (same approach as PyboostRunner constructor).
inline mindspore::device::DeviceContext *GetCurrentDeviceContext() {
  auto device_type = mindspore::DeviceManagerConf::GetInstance()->device_type();
  return mindspore::runtime::OpRunner::GetDeviceContext(device_type);
}

// ============================================================================
// TensorToDevice — prepare input tensors on the current device
// ============================================================================
namespace detail {

// Prepare one TensorToDevice(...) argument and return the next tensor-input index. Tensor-like arguments consume one
// input index and, if defined, create an input device address and trigger input malloc/lazy copy. Non-tensor arguments
// are ignored and do not consume input_index.
inline size_t CreateInputTensorAddressForArg(const mindspore::device::DeviceContext *dc, size_t stream_id,
                                             size_t input_index, const ms::Tensor &t) {
  if (t.is_defined()) {
    mindspore::pyboost::CreateInputTensorAddress(dc, stream_id, input_index, t.tensor());
  }
  return input_index + 1;
}

inline size_t CreateInputTensorAddressForArg(const mindspore::device::DeviceContext *dc, size_t stream_id,
                                             size_t input_index, const std::optional<ms::Tensor> &t) {
  if (t.has_value() && t->is_defined()) {
    std::optional<mindspore::tensor::TensorPtr> opt_ptr(t->tensor());
    mindspore::pyboost::CreateInputTensorAddress(dc, stream_id, input_index, opt_ptr);
  }
  return input_index + 1;
}

inline size_t CreateInputTensorAddressForArg(const mindspore::device::DeviceContext *dc, size_t stream_id,
                                             size_t input_index, const std::vector<ms::Tensor> &tensors) {
  std::vector<mindspore::tensor::TensorPtr> ptrs;
  ptrs.reserve(tensors.size());
  for (const auto &t : tensors) {
    if (t.is_defined()) {
      ptrs.push_back(t.tensor());
    }
  }
  mindspore::pyboost::CreateInputTensorAddress(dc, stream_id, input_index, ptrs);
  return input_index + 1;
}

// No-op for non-tensor types (scalars, enums, etc.)
template <typename T, std::enable_if_t<!std::is_same_v<std::decay_t<T>, ms::Tensor> &&
                                         !std::is_same_v<std::decay_t<T>, std::optional<ms::Tensor>> &&
                                         !std::is_same_v<std::decay_t<T>, std::vector<ms::Tensor>>,
                                       int> = 0>
size_t CreateInputTensorAddressForArg(const mindspore::device::DeviceContext *, size_t, size_t input_index, const T &) {
  return input_index;
}

}  // namespace detail

/// @brief Prepare tensor inputs on the current device before kernel launch.
///        Tensor-like arguments create input device addresses and dispatch input malloc/lazy copy.
///        Non-tensor arguments are skipped. Output tensors are handled by TensorAllocate.
/// @code
///   ms::TensorToDevice(x, y);           // two tensors
///   ms::TensorToDevice(x, type_id, y);  // tensor, scalar (skipped), tensor
/// @endcode
template <typename... Args>
void TensorToDevice(const Args &...args) {
  auto *dc = GetCurrentDeviceContext();
  auto stream_id = static_cast<size_t>(mindspore::CurrentStream::id());
  size_t input_index = 0;
  ((input_index = detail::CreateInputTensorAddressForArg(dc, stream_id, input_index, args)), ...);
}

// ============================================================================
// TensorAllocate — allocate device memory for output tensors
// ============================================================================

/// @brief Allocate device memory for a single output tensor.
PYBOOST_API void TensorAllocate(const ms::Tensor &output);

/// @brief Allocate device memory for a vector of output tensors.
PYBOOST_API void TensorAllocate(const std::vector<ms::Tensor> &outputs);

namespace detail {

inline void CollectOutputs(std::vector<mindspore::tensor::TensorPtr> &vec, const ms::Tensor &t) {
  if (t.is_defined()) {
    vec.push_back(t.tensor());
  }
}

inline void CollectOutputs(std::vector<mindspore::tensor::TensorPtr> &vec, const std::vector<ms::Tensor> &ts) {
  for (const auto &t : ts) {
    if (t.is_defined()) {
      vec.push_back(t.tensor());
    }
  }
}

}  // namespace detail

/// @brief Variadic TensorAllocate for multiple output tensors.
/// @code
///   ms::TensorAllocate(out1, out2);
/// @endcode
template <typename T1, typename T2, typename... Rest>
void TensorAllocate(const T1 &first, const T2 &second, const Rest &...rest) {
  auto *dc = GetCurrentDeviceContext();
  auto stream_id = static_cast<size_t>(mindspore::CurrentStream::id());
  std::vector<mindspore::tensor::TensorPtr> outputs;
  detail::CollectOutputs(outputs, first);
  detail::CollectOutputs(outputs, second);
  (detail::CollectOutputs(outputs, rest), ...);
  if (!outputs.empty()) {
    mindspore::pyboost::CreateOutputTensorAddress(dc, stream_id, outputs);
  }
}

// ============================================================================
// DispatchKernel — dispatch a CPU kernel via device task + launch task pipeline
// ============================================================================

/// @brief Dispatch a CPU kernel function asynchronously.
/// Matches the PyboostRunner two-level dispatch: device thread -> launch thread.
/// @code
///   ms::DispatchKernel([x, y, out]() {
///     auto *xp = x.data_ptr<float>();
///     // ... kernel logic ...
///   });
/// @endcode
template <typename Func>
void DispatchKernel(Func &&kernel_func) {
  auto *dc = GetCurrentDeviceContext();
  auto func_ptr = std::make_shared<std::decay_t<Func>>(std::forward<Func>(kernel_func));
  mindspore::kernel::pyboost::PyBoostUtils::DispatchRun(
    std::make_shared<mindspore::runtime::PyBoostDeviceTask>([dc, func_ptr]() {
      mindspore::runtime::OpExecutor::DispatchLaunchTask([dc, func_ptr]() {
        dc->device_res_manager_->BindDeviceToCurrentThread(false);
        (*func_ptr)();
      });
    }));
}

}  // namespace ms
#endif  // MINDSPORE_CCSRC_EXTENSION_CUSTOM_OP_API_H_
