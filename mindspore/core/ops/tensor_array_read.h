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

#ifndef MINDSPORE_CORE_OPS_TENSOR_ARRAY_READ_H_
#define MINDSPORE_CORE_OPS_TENSOR_ARRAY_READ_H_
#include <vector>
#include <string>
#include "ops/primitive_c.h"

namespace mindspore {
namespace ops {

constexpr auto kNameTensorArrayRead = "TensorArrayRead";

/// \brief Assert defined TensorArrayRead operator prototype of lite.
class MS_CORE_API TensorArrayRead : public PrimitiveC {
 public:
  /// \brief Constructor.
  TensorArrayRead() : PrimitiveC(kNameTensorArrayRead) { InitIOName({"handle", "index", "flow_in"}, {"tensor"}); }
  /// \brief Destructor.
  ~TensorArrayRead() = default;
  MS_DECLARE_PARENT(TensorArrayRead, PrimitiveC);
  /// \brief Method to init the op's attributes.
  void Init() {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TENSOR_ARRAY_READ_H_
