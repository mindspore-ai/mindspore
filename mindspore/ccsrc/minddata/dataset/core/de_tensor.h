/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DETENSOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DETENSOR_H_
#include <string>
#include <vector>
#include <memory>
#include "include/api/status.h"
#include "include/api/types.h"
#include "mindspore/core/ir/api_tensor_impl.h"
#include "minddata/dataset/core/tensor.h"

namespace mindspore {
namespace dataset {
class DETensor : public mindspore::MSTensor::Impl {
 public:
  DETensor() = default;
  ~DETensor() override = default;
  explicit DETensor(std::shared_ptr<dataset::Tensor> tensor_impl);
#ifndef ENABLE_ANDROID
  explicit DETensor(std::shared_ptr<dataset::DeviceTensor> device_tensor_impl, bool is_device);
#endif
  const std::string &Name() const override;

  enum mindspore::DataType DataType() const override;

  size_t DataSize() const override;

  const std::vector<int64_t> &Shape() const override;

  std::shared_ptr<const void> Data() const override;

  void *MutableData() override;

  bool IsDevice() const override;

  std::shared_ptr<mindspore::MSTensor::Impl> Clone() const override;

 private:
  std::shared_ptr<dataset::Tensor> tensor_impl_;
#ifndef ENABLE_ANDROID
  std::shared_ptr<dataset::DeviceTensor> device_tensor_impl_;
#endif
  bool is_device_;
  std::string name_;
  enum mindspore::DataType type_;
  std::vector<int64_t> shape_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DETENSOR_H_
