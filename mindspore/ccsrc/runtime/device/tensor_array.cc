/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
bool TensorArray::CheckValue(const TypeId &dtype, const ShapeVector &shape) {
  MS_LOG(DEBUG) << "Check the data shape and type for " << name_;
  MS_EXCEPTION_IF_NULL(dtype_);
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

// Add tensor to the TensorArray and increase the size.
// Cast 1: is_dynamic = False and index > max_size_, error.
// Case 2: index > valid_size, fill the rest dev_value with zeros, and set valid_size to index + 1.
// Case 3: index == tensors_.size(), we need to increase both real tensors_ size and valid size, and add
// the new dev_value to tensors_.
// Case 4: tensors_size() > index > valid_size, we can reuse the memory in tensors_[index], so
// only increase the valid_size.
bool TensorArray::Write(const int64_t index, const mindspore::kernel::AddressPtr &dev_value) {
  MS_LOG(DEBUG) << "Write dev_value to " << name_;
  if (!is_dynamic_ && (index >= max_size_)) {
    MS_LOG(ERROR) << name_ << " is not in dynamic size, the max_size is " << max_size_ << ", but get index " << index;
    return false;
  }
  if (LongToSize(index) > valid_size_) {
    // Create/reuse (index - valid_size) size dev_value with zeros.
    // 1 create new mem : index > real_size ? index - real_size : 0
    // 2 reuse old mem : index > real_size ? real_size - valid_size : index - valid_size
    // 3 fill zeros : index - valid_size
    MS_EXCEPTION_IF_NULL(dev_value);
    size_t create_size = (LongToSize(index) > tensors_.size()) ? (LongToSize(index) - tensors_.size()) : 0;
    for (size_t i = 0; i < create_size; i++) {
      kernel::AddressPtr create_dev = std::make_shared<kernel::Address>();
      create_dev->addr = AllocateMemory(dev_value->size);
      create_dev->size = dev_value->size;
      tensors_.push_back(create_dev);
    }
    tensors_.push_back(dev_value);
    for (size_t i = valid_size_; i < LongToSize(index); i++) {
      MS_EXCEPTION_IF_CHECK_FAIL((tensors_.size() > i), "The index is out of range.");
      MS_EXCEPTION_IF_NULL(tensors_[i]);
      auto tensor_size = tensors_[i]->size;
      ClearMemory(tensors_[i]->addr, tensor_size);
    }
    valid_size_ = LongToSize(index) + 1;
  } else if (LongToSize(index) == tensors_.size()) {
    MS_LOG(DEBUG) << "Write to index " << index << ", increase tensors' size to " << (tensors_.size() + 1);
    tensors_.push_back(dev_value);
    valid_size_++;
  } else {
    MS_LOG(DEBUG) << "Reuse tensors in position " << index << ", tensors size is " << tensors_.size();
    if (LongToSize(index) == valid_size_) {
      valid_size_++;
    }
  }
  return true;
}

void TensorArray::Clear() {
  valid_size_ = 0;
  return;
}

void TensorArray::Free() {
  MS_LOG(DEBUG) << "Free device memory for " << name_;
  for (const auto &addr : tensors_) {
    if (addr != nullptr) {
      FreeMemory(static_cast<DeviceMemPtr>(addr->addr));
    }
  }
}

size_t TensorArray::GetValidSize() const { return valid_size_; }
size_t TensorArray::GetRealSize() const { return tensors_.size(); }

const void *TensorArray::GetTensorAddr(const size_t &index) const {
  MS_EXCEPTION_IF_CHECK_FAIL((tensors_.size() > index), "The index is out of range.");
  MS_EXCEPTION_IF_NULL(tensors_[index]);
  return tensors_[index]->addr;
}

void TensorArray::SetMaxSize(const int64_t size, const bool is_dynamic) {
  is_dynamic_ = is_dynamic;
  if (!is_dynamic_) {
    max_size_ = size;
    MS_LOG(DEBUG) << name_ << " use fixed size " << max_size_;
  }
  return;
}
}  // namespace device
}  // namespace mindspore
