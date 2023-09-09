/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DEVICE_TENSOR_ASCEND910B_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DEVICE_TENSOR_ASCEND910B_H_

#include <memory>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/util/status.h"
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace dataset {
class Tensor;
class DATASET_API DeviceTensorAscend910B {
 public:
  DeviceTensorAscend910B(const TensorShape &shape, const DataType &type, device::DeviceContext *device_context,
                         const size_t &stream_id, bool is_hwc = true);

  // create device_tensor by empty
  static Status CreateDeviceTensor(const TensorShape &shape, const DataType &type,
                                   device::DeviceContext *device_context, const size_t &stream_id,
                                   std::shared_ptr<DeviceTensorAscend910B> *out, bool is_hwc = true);

  // create device_tensor by host tensor
  static Status CreateDeviceTensor(std::shared_ptr<Tensor> tensor, device::DeviceContext *device_context,
                                   const size_t &stream_id, std::shared_ptr<DeviceTensorAscend910B> *out,
                                   bool is_hwc = true);

  ~DeviceTensorAscend910B();

  device::DeviceContext *GetDeviceContext() { return device_context_; }

  size_t GetStreamID() { return stream_id_; }

  void SetDeviceAddress(void *device_address) { device_address_ = device_address; }

  void *GetDeviceAddress() { return device_address_; }

  void SetDeviceTensor(void *tensor) { tensor_ = tensor; }

  TensorShape &GetShape() { return tensor_shape_; }

  DataType GetType() { return data_type_; }

  void *GetDeviceTensor() { return tensor_; }

  Status ToHostTensor(std::shared_ptr<Tensor> *host_tensor);

 private:
  // Ascend910B resource
  device::DeviceContext *device_context_;
  size_t stream_id_;
  void *device_address_;
  void *tensor_;  // aclTensor
  TensorShape tensor_shape_;
  DataType data_type_;
  bool is_hwc_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DEVICE_TENSOR_ASCEND910B_H_
