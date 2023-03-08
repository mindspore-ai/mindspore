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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSOR_ARRAY_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSOR_ARRAY_H_

#include <vector>
#include <string>
#include <memory>
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/kernel.h"
#include "backend/common/mem_reuse/mem_dynamic_allocator.h"

namespace mindspore {
namespace device {
class BACKEND_EXPORT TensorArray {
 public:
  // Base TensorArray. Constructed by name, dtype and shapes.
  TensorArray(const string &name, const TypePtr &dtype, const ShapeVector &shapes)
      : name_(name), dtype_(dtype), shapes_(shapes), valid_size_(0), max_size_(0), is_dynamic_(true) {}
  virtual ~TensorArray() = default;

  // Check the index in valid range. Used in Read().
  virtual bool CheckReadIndexLogical(const int64_t index);
  // Check the dtype and shape of the input data. Used in Write().
  virtual bool CheckValue(const TypeId &dtype, const ShapeVector &shape);

  // Function Write() is used to insert or append dev_value to the position of index.
  virtual bool Write(const int64_t index, const mindspore::kernel::AddressPtr &dev_value);

  // Function Read() can get the tensors in the scope of tensors_.
  virtual mindspore::kernel::AddressPtr Read(const int64_t index);

  // Function Free() will release the memory in TensorArray.
  virtual void Free();

  // These three func should by implied for different device due to the difference in memory usage.
  // Create/Release Memory is used for malloc/free a device memory, used in function Write().
  // ClearMemory is used to reset the input addr with zeros, used in function Free().
  virtual void FreeMemory(const DeviceMemPtr addr) = 0;
  virtual void *AllocateMemory(const size_t size) = 0;
  virtual void ClearMemory(void *addr, const size_t size) = 0;

  // Clear() will only set the valid size of TensorArray to zero. The memory in TensorArray is still
  // kept. In this situation， we can reuse the memory for next using.
  virtual void Clear();

  // A vector of tensor address are kept in a TensorArray. For memory reusing, we will keep the addr
  // after Clear(), in this time, the valid size will be zero but the real size still kept as
  // tensors_.size(). Overall， using GetValidSize() to get a logical TensorArray size, and using
  // GetRealSize() to get a physical TensorArray size.
  virtual size_t GetValidSize() const;
  virtual size_t GetRealSize() const;

  // This function is used in the situation that is_dynamic == false then set the max size.
  // Otherwise, it won't be used and use the default implement.
  virtual void SetMaxSize(const int64_t size, const bool is_dynamic);

  // Return the tensor address in position index.
  virtual const void *GetTensorAddr(const size_t &index) const;

 protected:
  std::string name_;
  TypePtr dtype_;
  ShapeVector shapes_;
  size_t valid_size_;
  int64_t max_size_;
  bool is_dynamic_;
  // Using a vector tensors_ to store the dev_tensor_addr from Write().
  std::vector<mindspore::kernel::AddressPtr> tensors_;
};
using TensorArrayPtr = std::shared_ptr<TensorArray>;
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSOR_ARRAY_H_
