/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_INCLUDE_MS_TENSOR_H_
#define MINDSPORE_INCLUDE_MS_TENSOR_H_

#include <utility>
#include <vector>
#include <memory>
#include "ir/dtype/type_id.h"

namespace mindspore {
#define MS_API __attribute__((visibility("default")))
namespace inference {
class MS_API MSTensor {
 public:
  MSTensor() = default;
  // brief Create a MSTensor pointer.
  //
  // param data_type DataTypeId of tensor to be created.
  // param shape Shape of tensor to be created.
  // return MSTensor pointer.
  static MSTensor *CreateTensor(TypeId data_type, const std::vector<int> &shape);

  ~MSTensor() = default;

  virtual TypeId data_type() const = 0;

  virtual TypeId set_data_type(const TypeId data_type) = 0;

  virtual std::vector<int> shape() const = 0;

  virtual size_t set_shape(const std::vector<int> &shape) = 0;

  virtual int DimensionSize(size_t index) const = 0;
  // brief Get number of element in MSTensor.
  //
  // return Number of element in MSTensor.
  virtual int ElementsNum() const = 0;

  virtual std::size_t hash() const = 0;
  // brief Get byte size of data in MSTensor.
  //
  // return Byte size of data in MSTensor.
  virtual size_t Size() const = 0;
  // brief Get pointer of data in MSTensor.
  //
  // The data pointer can be used to both write or read data in MSTensor.
  //
  // return A pointer points to data in MSTensor.
  virtual void *MutableData() const = 0;
};
using MultiTensor = std::vector<std::vector<std::shared_ptr<inference::MSTensor>>>;
}  // namespace inference
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_MS_TENSOR_H_
