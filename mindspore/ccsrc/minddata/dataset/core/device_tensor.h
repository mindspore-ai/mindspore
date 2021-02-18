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
#include "minddata/dataset/core/constants.h"
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

  uint8_t *GetDeviceBuffer();

  std::vector<uint32_t> GetYuvStrideShape();

  uint32_t DeviceDataSize();

  bool HasDeviceData() { return device_data_ != nullptr; }

 private:
  Status SetSize_(const uint32_t &new_size);

  Status SetYuvStrideShape_(const uint32_t &width, const uint32_t &widthStride, const uint32_t &height,
                            const uint32_t &heightStride);

  std::vector<uint32_t> YUV_shape_;
  uint8_t *device_data_;
  uint32_t size_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DEVICE_TENSOR_H_
