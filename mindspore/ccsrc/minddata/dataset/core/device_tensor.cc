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
#ifdef ENABLE_ACL
#include "minddata/dataset/kernels/image/dvpp/utils/MDAclProcess.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
DeviceTensor::DeviceTensor(const TensorShape &shape, const DataType &type) : Tensor(shape, type) {
  // grab the mem pool from global context and create the allocator for char data area
  std::shared_ptr<MemoryPool> global_pool = GlobalContext::Instance()->mem_pool();
  data_allocator_ = std::make_unique<Allocator<unsigned char>>(global_pool);
  device_data_type_ = type;
  host_data_tensor_ = nullptr;
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

Status DeviceTensor::CreateFromDeviceMemory(const TensorShape &shape, const DataType &type, uint8_t *data_ptr,
                                            const uint32_t &dataSize, const std::vector<uint32_t> &attributes,
                                            std::shared_ptr<DeviceTensor> *out) {
  CHECK_FAIL_RETURN_UNEXPECTED(shape.known(), "Invalid shape.");
  CHECK_FAIL_RETURN_UNEXPECTED(type != DataType::DE_UNKNOWN, "Invalid data type.");
  CHECK_FAIL_RETURN_UNEXPECTED(data_ptr != nullptr, "Data pointer is NULL");
  CHECK_FAIL_RETURN_UNEXPECTED(dataSize > 0, "Invalid data size");

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

  CHECK_FAIL_RETURN_UNEXPECTED(
    (*out)->SetAttributes(data_ptr, dataSize, attributes[0], attributes[1], attributes[2], attributes[3]),
    "Fail to set attributes for DeviceTensor");

  return Status::OK();
}

const unsigned char *DeviceTensor::GetHostBuffer() {
#ifdef ENABLE_ACL
  Status rc = DataPop_(&host_data_tensor_);
  if (!rc.IsOk()) {
    MS_LOG(ERROR) << "Pop device data onto host fail, a nullptr will be returned";
    return nullptr;
  }
#endif
  if (!host_data_tensor_) {
    return nullptr;
  }
  return host_data_tensor_->GetBuffer();
}

const uint8_t *DeviceTensor::GetDeviceBuffer() { return device_data_; }

uint8_t *DeviceTensor::GetDeviceMutableBuffer() { return device_data_; }

DataType DeviceTensor::DeviceDataType() const { return device_data_type_; }

uint32_t DeviceTensor::DeviceDataSize() { return size_; }

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

Status DeviceTensor::SetSize_(const uint32_t &new_size) {
  size_ = new_size;
  return Status::OK();
}

#ifdef ENABLE_ACL
Status DeviceTensor::DataPop_(std::shared_ptr<Tensor> *host_tensor) {
  void *resHostBuf = nullptr;
  APP_ERROR ret = aclrtMallocHost(&resHostBuf, this->DeviceDataSize());
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to allocate memory from host ret = " << ret;
    return Status(StatusCode::kMDNoSpace);
  }

  std::shared_ptr<void> outBuf(resHostBuf, aclrtFreeHost);
  auto processedInfo_ = outBuf;
  // Memcpy the output data from device to host
  ret = aclrtMemcpy(outBuf.get(), this->DeviceDataSize(), this->GetDeviceBuffer(), this->DeviceDataSize(),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy memory from device to host, ret = " << ret;
    return Status(StatusCode::kMDOutOfMemory);
  }

  auto data = std::static_pointer_cast<unsigned char>(processedInfo_);
  unsigned char *ret_ptr = data.get();

  mindspore::dataset::dsize_t dvppDataSize = this->DeviceDataSize();
  const mindspore::dataset::TensorShape dvpp_shape({dvppDataSize, 1, 1});
  uint32_t _output_width_ = this->GetYuvStrideShape()[0];
  uint32_t _output_widthStride_ = this->GetYuvStrideShape()[1];
  uint32_t _output_height_ = this->GetYuvStrideShape()[2];
  uint32_t _output_heightStride_ = this->GetYuvStrideShape()[3];
  const mindspore::dataset::DataType dvpp_data_type(mindspore::dataset::DataType::DE_UINT8);

  mindspore::dataset::Tensor::CreateFromMemory(dvpp_shape, dvpp_data_type, ret_ptr, host_tensor);

  (*host_tensor)->SetYuvShape(_output_width_, _output_widthStride_, _output_height_, _output_heightStride_);
  if (!(*host_tensor)->HasData()) {
    return Status(StatusCode::kMCDeviceError);
  }

  MS_LOG(INFO) << "Successfully pop DeviceTensor data onto host";
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
