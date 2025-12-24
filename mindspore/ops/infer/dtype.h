/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_DTYPE_H_
#define MINDSPORE_CORE_OPS_DTYPE_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDType = "DType";
constexpr auto kNameDTypeId = "DTypeId";

/// \brief Returns the data type of the input tensor as mindspore.dtype.
/// Refer to Python API @ref mindspore.ops.DType for more details.
class OPS_API DType : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DType);
  /// \brief Constructor.
  DType() : BaseOperator(kNameDType) { InitIOName({"x"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.DType for the inputs.
  void Init() const {}
};

/// \brief Returns the data type of the input tensor as mindspore.dtype.
/// Refer to Python API @ref mindspore.ops.DType for more details.
class OPS_API DTypeId : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DTypeId);
  /// \brief Constructor.
  DTypeId() : BaseOperator(kNameDTypeId) { InitIOName({"x"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.DType for the inputs.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DTYPE_H_
