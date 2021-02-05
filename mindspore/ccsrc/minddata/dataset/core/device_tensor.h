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

#ifndef MINDSPORE_DEVICE_TENSOR_H
#define MINDSPORE_DEVICE_TENSOR_H
#include <memory>
#include <utility>
#include <vector>
#include "include/api/status.h"
#include "minddata/dataset/core/tensor.h"
#ifdef ENABLE_ACL
#include "minddata/dataset/kernels/image/dvpp/utils/DvppCommon.h"
#endif
#include "minddata/dataset/core/constants.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class Tensor;
class DeviceTensor : public Tensor {
 public:
  DeviceTensor(const TensorShape &shape, const DataType &type);

  ~DeviceTensor() {}
#ifdef ENABLE_ACL
  Status SetAttributes(const std::shared_ptr<DvppDataInfo> &data);
#endif
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
#endif  // MINDSPORE_DEVICE_TENSOR_H
