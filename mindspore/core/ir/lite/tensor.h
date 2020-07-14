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

#ifndef MINDSPORE_CCSRC_IR_LITE_TENSOR_H_
#define MINDSPORE_CCSRC_IR_LITE_TENSOR_H_

#include <memory>
#include <vector>
#include "ir/meta_tensor.h"
#include "ir/dtype/type.h"

namespace mindspore {
namespace tensor {
class Tensor : public MetaTensor {
 public:
  Tensor() : MetaTensor() {}

  Tensor(const TypeId data_type, const std::vector<int> &shape);

  Tensor(const TypePtr &type_ptr, const std::vector<int> &shape);

  Tensor(const Tensor &tensor);

  ~Tensor();

  int CopyTensorData(const Tensor &srcTensor);

  MS_DECLARE_PARENT(Tensor, MetaTensor)

  virtual Tensor &operator=(const Tensor &tensor);

  virtual bool operator==(const Tensor &tensor);

  bool operator==(const Value &other) const override;

  size_t Size() const { return MetaTensor::ElementsNum() * GetTypeByte(TypeIdToType(this->data_type_)); }

  void *Data() const { return data_; }

 protected:
  void *data_;
};

using TensorPtr = std::shared_ptr<Tensor>;
}  // namespace tensor

namespace inference {
class Tensor : public MSTensor {
 public:
  Tensor();

  Tensor(TypeId data_type, const std::vector<int> &shape);

  explicit Tensor(std::shared_ptr<tensor::Tensor> tensor_ptr);

  ~Tensor() = default;

  TypeId data_type() const override;

  TypeId set_data_type(const TypeId data_type) override;

  std::vector<int> shape() const override;

  size_t set_shape(const std::vector<int> &shape) override;

  int DimensionSize(size_t index) const override;

  int ElementsNum() const override;

  std::size_t hash() const override;

  std::shared_ptr<tensor::Tensor> tensor() const;

  size_t Size() const override;

  void *MutableData() const override;

 protected:
  std::shared_ptr<tensor::Tensor> tensor_impl_;
};
}  // namespace inference
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_LITE_TENSOR_H_
