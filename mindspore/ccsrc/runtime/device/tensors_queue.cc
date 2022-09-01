/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <functional>
#include "runtime/device/tensors_queue.h"

namespace mindspore {
namespace device {
void TensorsQueue::CreateTensorsQueue() {
  // Store one element tensors' size.
  // The whole TensorsQueue is like: [[tensor1, tensor2], [tensor3, tensor4]].
  // One element means [tensor1, tensor2].
  std::vector<int64_t> element_size_list;
  for (auto shape : shapes_) {
    int64_t item_size =
      std::accumulate(shape.begin(), shape.end(), SizeToLong(GetTypeByte(dtype_)), std::multiplies<int64_t>());
    element_size_list.push_back(item_size);
  }
  // Create the elements in TensorsQueue when construct.
  for (int64_t i = 0; i < size_; i++) {
    mindspore::kernel::AddressPtrList element_addrs;
    for (auto element_size : element_size_list) {
      kernel::AddressPtr create_dev = std::make_shared<kernel::Address>();
      create_dev->addr = AllocateMemory(LongToSize(element_size));
      create_dev->size = LongToSize(element_size);
      element_addrs.push_back(create_dev);
      MS_LOG(DEBUG) << "Create  " << element_size << "bytes for " << name_;
    }
    tensors_q_.push_back(element_addrs);
  }
  MS_LOG(DEBUG) << "Create a TensorsQueue: " << name_ << ", Q size is " << size_ << ", elements num is "
                << elements_num_;
}

void TensorsQueue::CopyTensor(const mindspore::kernel::AddressPtr &, const mindspore::kernel::AddressPtr &) {
  MS_LOG(EXCEPTION) << "This should be overridden by subclass !";
}
void TensorsQueue::CopyTensor(const mindspore::kernel::AddressPtr &, const mindspore::kernel::AddressPtr &, void *) {
  MS_LOG(EXCEPTION) << "This should be overridden by subclass !";
}

size_t TensorsQueue::AvailableSize() {
  return (rear_ > front_) ? (rear_ - front_) : (LongToSize(size_) - front_ + rear_);
}
bool TensorsQueue::IsFull() {
  if (size_ <= 0) {
    return false;
  } else {
    return (rear_ + IntToSize(1)) % LongToSize(size_) == front_;
  }
}
bool TensorsQueue::IsEmpty() { return front_ == rear_; }

bool TensorsQueue::Put(const mindspore::kernel::AddressPtrList &dev_addr) {
  // When the tensor_q is full, put will failed.
  if (IsFull()) {
    MS_LOG(WARNING) << "The " << name_ << " is full, total size is " << size_;
    return false;
  }
  // Get the element in position rear_ and change the value by input, the we increase the rear_.
  // We can get a effect like a circle queue and reuse the addrs.
  MS_EXCEPTION_IF_CHECK_FAIL((tensors_q_.size() > rear_), "The index is out of range.");
  mindspore::kernel::AddressPtrList element = tensors_q_[rear_];
  for (int64_t i = 0; i < elements_num_; i++) {
    CopyTensor(element[LongToSize(i)], dev_addr[LongToSize(i) + IntToSize(1)]);
  }
  if (size_ <= 0) {
    return false;
  }
  rear_ = (rear_ + 1) % LongToSize(size_);
  MS_LOG(DEBUG) << "Put an element into  " << name_ << ", now the avliable q size is [" << AvailableSize() << "/"
                << size_ << "]";
  return true;
}

bool TensorsQueue::Put(const mindspore::kernel::AddressPtrList &dev_addr, void *stream) {
  if (IsFull()) {
    MS_LOG(WARNING) << "The " << name_ << " is full, total size is " << size_;
    return false;
  }
  MS_EXCEPTION_IF_CHECK_FAIL((tensors_q_.size() > rear_), "The index is out of range.");
  mindspore::kernel::AddressPtrList element = tensors_q_[rear_];
  for (int64_t i = 0; i < elements_num_; i++) {
    CopyTensor(element[LongToSize(i)], dev_addr[LongToSize(i) + IntToSize(1)], stream);
  }
  if (size_ <= 0) {
    return false;
  }
  rear_ = (rear_ + IntToSize(1)) % LongToSize(size_);
  MS_LOG(DEBUG) << "Put an element into  " << name_ << ", now the avliable q size is [" << AvailableSize() << "/"
                << size_ << "]";
  return true;
}

bool TensorsQueue::Get(const mindspore::kernel::AddressPtrList &dev_addr, const bool &pop_after_get, void *stream) {
  // Get a tensor addrs list from the queue.
  // If pop_after_get is true, we will pop the addrs from tensors_q_.
  if (IsEmpty()) {
    MS_LOG(WARNING) << "The TensorsQueue " << name_ << " is empty";
    return false;
  }
  MS_EXCEPTION_IF_CHECK_FAIL((tensors_q_.size() > front_), "The index is out of range.");
  mindspore::kernel::AddressPtrList element = tensors_q_[front_];
  for (int64_t i = 0; i < elements_num_; i++) {
    CopyTensor(dev_addr[LongToSize(i)], element[LongToSize(i)], stream);
  }
  if (pop_after_get) {
    if (size_ <= 0) {
      MS_LOG(ERROR) << "The size is zero.";
      return false;
    }
    front_ = (front_ + IntToSize(1)) % LongToSize(size_);
  }
  MS_LOG(DEBUG) << "Get an element from  " << name_ << ", pop_after_get is " << pop_after_get
                << ", now the avliable q size is[" << AvailableSize() << " / " << size_ << "] ";
  return true;
}

bool TensorsQueue::Get(const mindspore::kernel::AddressPtrList &dev_addr, const bool &pop_after_get) {
  if (IsEmpty()) {
    MS_LOG(WARNING) << "The TensorsQueue " << name_ << " is empty";
    return false;
  }
  MS_EXCEPTION_IF_CHECK_FAIL((tensors_q_.size() > front_), "The index is out of range.");
  mindspore::kernel::AddressPtrList element = tensors_q_.front();
  for (int64_t i = 0; i < elements_num_; i++) {
    CopyTensor(dev_addr[LongToSize(i)], element[LongToSize(i)]);
  }
  if (pop_after_get) {
    if (size_ <= 0) {
      MS_LOG(ERROR) << "The size is zero.";
      return false;
    }
    front_ = (front_ + IntToSize(1)) % LongToSize(size_);
  }
  MS_LOG(DEBUG) << "Get an element from  " << name_ << ", pop_after_get is " << pop_after_get
                << ", now the avliable q size is[" << AvailableSize() << " / " << size_ << "] ";
  return true;
}

void TensorsQueue::Clear() {
  // Clear the tensors_q_ and return the element addr back to tensors_store.
  if (IsEmpty()) {
    MS_LOG(WARNING) << "The TensorsQueue " << name_ << " is already empty when execute Clear.";
  }
  rear_ = 0;
  front_ = 0;
  MS_LOG(DEBUG) << "Clear the elements for " << name_;
}

void TensorsQueue::Free() {
  while (!IsEmpty()) {
    MS_EXCEPTION_IF_CHECK_FAIL((tensors_q_.size() > front_), "The index is out of range.");
    auto element = tensors_q_[front_];
    for (const auto &addr : element) {
      if (addr != nullptr) {
        FreeMemory(static_cast<DeviceMemPtr>(addr->addr));
      }
    }

    if (size_ <= 0) {
      MS_LOG(ERROR) << "The size is zero.";
      return;
    }
    front_ = (front_ + IntToSize(1)) % LongToSize(size_);
  }
  MS_LOG(DEBUG) << "Free the TensorsQueue's memory for " << name_;
}
}  // namespace device
}  // namespace mindspore
