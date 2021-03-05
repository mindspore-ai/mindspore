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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DEVICE_TENSOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DEVICE_TENSOR_H_
#include <memory>
#include <utility>
#include <vector>
#include "include/api/status.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class Tensor;
class DeviceTensor : public Tensor {
 public:
  DeviceTensor(const TensorShape &shape, const DataType &type);

  ~DeviceTensor() {}

  Status SetAttributes(uint8_t *data_ptr, const uint32_t &dataSize, const uint32_t &width, const uint32_t &widthStride,
                       const uint32_t &height, const uint32_t &heightStride);

  static Status CreateEmpty(const TensorShape &shape, const DataType &type, std::shared_ptr<DeviceTensor> *out);

  static Status CreateFromDeviceMemory(const TensorShape &shape, const DataType &type, uint8_t *data_ptr,
                                       const uint32_t &dataSize, const std::vector<uint32_t> &attributes,
                                       std::shared_ptr<DeviceTensor> *out);

  const unsigned char *GetHostBuffer();

  const uint8_t *GetDeviceBuffer();

  uint8_t *GetDeviceMutableBuffer();

  std::vector<uint32_t> GetYuvStrideShape();

  uint32_t DeviceDataSize();

  DataType DeviceDataType() const;

  bool HasDeviceData() { return device_data_ != nullptr; }

 private:
  Status SetSize_(const uint32_t &new_size);

  Status SetYuvStrideShape_(const uint32_t &width, const uint32_t &widthStride, const uint32_t &height,
                            const uint32_t &heightStride);

#ifdef ENABLE_ACL
  Status DataPop_(std::shared_ptr<Tensor> *host_tensor);
#endif

  std::vector<uint32_t> YUV_shape_;  // YUV_shape_ = {width, widthStride, height, heightStride}

  uint8_t *device_data_;

  uint32_t size_;

  DataType device_data_type_;

  // We use this Tensor to store device_data when DeviceTensor pop onto host
  std::shared_ptr<Tensor> host_data_tensor_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DEVICE_TENSOR_H_
