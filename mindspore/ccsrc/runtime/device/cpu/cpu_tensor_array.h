/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_TENSOR_ARRAY_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_TENSOR_ARRAY_H_

#include <vector>
#include <string>
#include <memory>
#include "runtime/device/tensor_array.h"

namespace mindspore {
namespace device {
namespace cpu {
class CPUTensorArray : public TensorArray {
 public:
  CPUTensorArray(const string &name, const TypePtr &dtype, const std::vector<size_t> &shapes)
      : TensorArray(name, dtype, shapes) {}
  ~CPUTensorArray() override = default;

  bool Write(const int64_t index, const mindspore::kernel::AddressPtr &dev_value) override;
  void Free() override;
  void Clear() override { valid_size_ = 0; }

  size_t GetValidSize() const override { return valid_size_; }
  size_t GetRealSize() const override { return tensors_.size(); }

  void *GetTensorAddr(const size_t &index) const { return tensors_[index]->addr; }

  void SetMaxSize(const int64_t size, const bool is_dynamic) override {
    is_dynamic_ = is_dynamic;
    if (!is_dynamic) {
      max_size_ = size;
    }
  }

 private:
  int64_t max_size_;
  bool is_dynamic_;
};
using CPUTensorArray = CPUTensorArray;
using CPUTensorArrayPtr = std::shared_ptr<CPUTensorArray>;
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_TENSOR_ARRAY_H_
