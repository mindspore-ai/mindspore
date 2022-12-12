/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_ADAPTER_TENSOR_H_
#define MINDSPORE_CORE_IR_ADAPTER_TENSOR_H_

#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/tensor.h"

namespace mindspore {
namespace tensor {
class AdapterTensor;
// Smart pointer for AdapterTensor.
using AdapterTensorPtr = std::shared_ptr<AdapterTensor>;
///
/// \brief AdapterTensor is used to map the Tensor of other frameworks.
///
class MS_CORE_API AdapterTensor final : public Tensor {
 public:
  /// \brief Create AdapterTensor from tensor.
  ///
  /// \param[in] tensor The input tensor.
  explicit AdapterTensor(const TensorPtr &tensor) : Tensor(*tensor), origin_tensor_(tensor) {}

  AdapterTensor() = default;

  /// Destructor of AdapterTensor.
  ~AdapterTensor() override = default;

  MS_DECLARE_PARENT(AdapterTensor, Tensor);

  bool operator==(const AdapterTensor &other) const;

  abstract::AbstractBasePtr ToAbstract() override;

 private:
  TensorPtr origin_tensor_{nullptr};
};
}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_ADAPTER_TENSOR_H_
