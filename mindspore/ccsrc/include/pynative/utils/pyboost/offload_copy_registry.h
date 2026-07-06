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

#ifndef MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_OFFLOAD_COPY_REGISTRY_H_
#define MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_OFFLOAD_COPY_REGISTRY_H_

#include <functional>
#include <utility>
#include "ir/tensor.h"
#include "mindapi/base/macros.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

// Function type for D2H copy: (stream_id, device_tensor) -> host_tensor
using CopyToHostFunc = std::function<tensor::TensorPtr(size_t, const tensor::TensorPtr &)>;

// Function type for H2D copy: (stream_id, host_tensor) -> device_tensor
using CopyToDeviceFunc = std::function<tensor::TensorPtr(size_t, const tensor::TensorPtr &)>;

// Registry for offload copy functions.
// This allows mindspore_ops_ascend to register implementations that can be called from mindspore_pynative.
class PYBOOST_API OffloadCopyRegistry {
 public:
  static OffloadCopyRegistry &GetInstance();

  // Register copy functions (called by mindspore_ops_ascend during static init)
  void RegisterCopyToHost(CopyToHostFunc func);
  void RegisterCopyToDevice(CopyToDeviceFunc func);

  // Check if functions are registered
  bool HasCopyToHost() const;
  bool HasCopyToDevice() const;

  // Call registered functions
  tensor::TensorPtr CopyToHost(size_t stream_id, const tensor::TensorPtr &device_tensor);

  tensor::TensorPtr CopyToDevice(size_t stream_id, const tensor::TensorPtr &host_tensor);

 private:
  OffloadCopyRegistry();
  ~OffloadCopyRegistry();
  DISABLE_COPY_AND_ASSIGN(OffloadCopyRegistry);

  CopyToHostFunc copy_to_host_func_{nullptr};
  CopyToDeviceFunc copy_to_device_func_{nullptr};
};

// Helper class for static registration
class OffloadCopyRegistrar {
 public:
  OffloadCopyRegistrar(CopyToHostFunc to_host, CopyToDeviceFunc to_device) {
    OffloadCopyRegistry::GetInstance().RegisterCopyToHost(std::move(to_host));
    OffloadCopyRegistry::GetInstance().RegisterCopyToDevice(std::move(to_device));
  }
};

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_OFFLOAD_COPY_REGISTRY_H_
