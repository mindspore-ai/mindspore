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

#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status DeviceTensor::SetYuvStrideShape_(const uint32_t &width, const uint32_t &widthStride, const uint32_t &height,
                                        const uint32_t &heightStride) {
  YUV_shape_ = {width, widthStride, height, heightStride};
  return Status::OK();
}

std::vector<uint32_t> DeviceTensor::GetYuvStrideShape() { return YUV_shape_; }

Status DeviceTensor::SetAttributes(uint8_t *data_ptr, const uint32_t &dataSize, const uint32_t &width,
                                   const uint32_t &widthStride, const uint32_t &height, const uint32_t &heightStride) {
  device_data_ = data_ptr;
  CHECK_FAIL_RETURN_UNEXPECTED(device_data_ != nullptr, "Fail to get the device data.");
  SetSize_(dataSize);
  SetYuvStrideShape_(width, widthStride, height, heightStride);
  return Status::OK();
}

DeviceTensor::DeviceTensor(const TensorShape &shape, const DataType &type) : Tensor(shape, type) {
  // grab the mem pool from global context and create the allocator for char data area
  std::shared_ptr<MemoryPool> global_pool = GlobalContext::Instance()->mem_pool();
  data_allocator_ = std::make_unique<Allocator<unsigned char>>(global_pool);
}

Status DeviceTensor::CreateEmpty(const TensorShape &shape, const DataType &type, std::shared_ptr<DeviceTensor> *out) {
  CHECK_FAIL_RETURN_UNEXPECTED(shape.known(), "Invalid shape.");
  CHECK_FAIL_RETURN_UNEXPECTED(type != DataType::DE_UNKNOWN, "Invalid data type.");
  const DeviceTensorAlloc *alloc = GlobalContext::Instance()->device_tensor_allocator();
  *out = std::allocate_shared<DeviceTensor>(*alloc, shape, type);
  // if it's a string tensor and it has no elements, Just initialize the shape and type.
  if (!type.IsNumeric() && shape.NumOfElements() == 0) {
    return Status::OK();
  }

  CHECK_FAIL_RETURN_UNEXPECTED(type.IsNumeric(), "Number of elements is not 0. The type should be numeric.");

  int64_t byte_size = (*out)->SizeInBytes();

  // Don't allocate if we have a tensor with no elements.
  if (byte_size != 0) {
    RETURN_IF_NOT_OK((*out)->AllocateBuffer(byte_size));
  }
  return Status::OK();
}

uint8_t *DeviceTensor::GetDeviceBuffer() { return device_data_; }

uint32_t DeviceTensor::DeviceDataSize() { return size_; }

Status DeviceTensor::SetSize_(const uint32_t &new_size) {
  size_ = new_size;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
