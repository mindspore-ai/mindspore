/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DE_TENSOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DE_TENSOR_H_

#include <memory>
#include <string>
#include <vector>

#include "utils/status.h"
#include "include/dataset/ms_tensor.h"

namespace mindspore::dataset {
class Tensor;
class DeviceTensor;

class DETensor : public mindspore::MSTensor::Impl {
 public:
  ~DETensor() = default;

  explicit DETensor(std::shared_ptr<dataset::Tensor> tensor_impl);

  DETensor(std::shared_ptr<dataset::DeviceTensor> device_tensor_impl, bool is_device);

  enum mindspore::DataType DataType() const override;

  size_t DataSize() const override;

  const std::vector<int64_t> &Shape() const override;

  std::shared_ptr<const void> Data() const override;

  void *MutableData() override;

 private:
  std::shared_ptr<dataset::Tensor> tensor_impl_;
  std::shared_ptr<dataset::DeviceTensor> device_tensor_impl_;
  bool is_device_;
  std::string name_;
  std::vector<int64_t> shape_;
};
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DE_TENSOR_H_
