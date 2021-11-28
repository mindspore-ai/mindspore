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

#include "runtime/device/cpu/cpu_tensor_array.h"
#include <vector>
#include <string>
#include <memory>
#include "runtime/hardware/cpu/cpu_memory_pool.h"

namespace mindspore {
namespace device {
namespace cpu {
// Add tensor to the TensorArray and increase the size.
// Cast 1: is_dynamic = False and index > max_size_, error.
// Case 2: index > valid_size, fill the rest dev_value with zeros, and set valid_size to index + 1.
// Case 3: index == tensors_.size(), we need to increase both real tensors_ size and valid size, and add
// the new dev_value to tensors_.
// Case 4: tensors_size() > index > valid_size, we can reuse the memory in tensors_[index], so
// only increase the valid_size.
bool CPUTensorArray::Write(const int64_t index, const mindspore::kernel::AddressPtr &dev_value) {
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
    size_t create_size = (LongToSize(index) > tensors_.size()) ? (LongToSize(index) - tensors_.size()) : 0;
    for (size_t i = 0; i < create_size; i++) {
      kernel::AddressPtr create_dev = std::make_shared<kernel::Address>();
      create_dev->addr = CPUMemoryPool::GetInstance().AllocTensorMem(dev_value->size);
      create_dev->size = dev_value->size;
      tensors_.push_back(create_dev);
    }
    tensors_.push_back(dev_value);
    // FillZeros(valid_size_, index);
    for (size_t i = valid_size_; i < LongToSize(index); i++) {
      auto tensor_size = tensors_[i]->size;
      (void)memset_s(tensors_[i]->addr, tensor_size, 0, tensors_[i]->size);
    }
    valid_size_ = LongToSize(index) + 1;
  } else if (LongToSize(index) == tensors_.size()) {
    MS_LOG(DEBUG) << "Write to index " << index << ", increase tensors' size to " << (tensors_.size() + 1);
    tensors_.push_back(dev_value);
    valid_size_++;
  } else {
    MS_LOG(DEBUG) << "Reuse tensors in position " << index << ", tensors size is " << tensors_.size();
    if (LongToSize(index) == valid_size_) valid_size_++;
  }
  return true;
}

// Free() will free the memory in TensorArray.
void CPUTensorArray::Free() {
  MS_LOG(DEBUG) << "Free device memory for " << name_;
  for (const auto &addr : tensors_) {
    if (addr != nullptr) {
      CPUMemoryPool::GetInstance().FreeTensorMem(static_cast<void *>(addr->addr));
    }
  }
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
