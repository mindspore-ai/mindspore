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
#include "runtime/device/tensor_array.h"

namespace mindspore {
namespace device {
bool TensorArray::CheckValue(const TypeId &dtype, const std::vector<size_t> &shape) {
  MS_LOG(DEBUG) << "Check the data shape and type for " << name_;
  if (dtype != dtype_->type_id()) {
    MS_LOG(ERROR) << "Invalid data type " << TypeIdLabel(dtype) << " for " << name_ << ", the origin type is "
                  << TypeIdLabel(dtype_->type_id());
    return false;
  }
  if (shape != shapes_) {
    MS_LOG(ERROR) << "Invalid data shape " << shape << " for " << name_ << ", the origin shape is " << shapes_;
    return false;
  }
  return true;
}

bool TensorArray::CheckReadIndexLogical(const int64_t index) {
  if (LongToSize(index) >= valid_size_) {
    MS_LOG(ERROR) << "Index " << index << " out of range " << valid_size_ << ", " << name_;
    return false;
  }
  return true;
}

// Function Read() can get the tensors in the scope of tensors_.
mindspore::kernel::AddressPtr TensorArray::Read(const int64_t index) {
  if (LongToSize(index) >= tensors_.size()) {
    MS_LOG(EXCEPTION) << "Index " << index << " out of range " << tensors_.size() << ", " << name_;
  }
  MS_LOG(DEBUG) << "Read tensor index = " << index << ", addr = " << tensors_[LongToSize(index)]->addr;
  return tensors_[LongToSize(index)];
}

void TensorArray::Clear() {
  valid_size_ = 0;
  return;
}

size_t TensorArray::GetRealSize() const { return valid_size_; }

void TensorArray::SetMaxSize(const int64_t size, const bool is_dynamic) {
  MS_LOG(DEBUG) << name_ << " use default SetTensorArrayMaxSize, and keep it empty";
  return;
}
}  // namespace device
}  // namespace mindspore
