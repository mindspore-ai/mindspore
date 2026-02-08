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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_PYBOOST_FUNCTIONS_COMPOSITE_TO_BASE_H_
#define MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_PYBOOST_FUNCTIONS_COMPOSITE_TO_BASE_H_
#include <utility>
#include <map>
#include "mindapi/base/types.h"
#include "include/pynative/utils/pyboost/functions/composite/empty_like.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
device::DeviceType GetDeviceType(Device device);
Int64ImmPtr GetDeviceByDeviceType(device::DeviceType device_type);

class ToProcessor {
 public:
  ToProcessor(TensorPtr tensor, device::DeviceType device_type, TypeId dtype, bool non_blocking, bool copy)
      : tensor_(std::move(tensor)), device_type_(device_type), dtype_(dtype), non_blocking_(non_blocking), copy_(copy) {
    origin_ = tensor_.get();
  }

  ToProcessor &Contiguous();

  ToProcessor &Cast();

  ToProcessor &Device();

  ToProcessor &Copy();

  const TensorPtr &Get() { return tensor_; }

 private:
  tensor::TensorPtr tensor_;
  device::DeviceType device_type_;
  TypeId dtype_;
  bool non_blocking_;
  bool copy_;
  void *origin_;
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_PYBOOST_FUNCTIONS_COMPOSITE_TO_BASE_H_
