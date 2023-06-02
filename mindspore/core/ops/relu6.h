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
#ifndef MINDSPORE_CORE_OPS_RELU6_H_
#define MINDSPORE_CORE_OPS_RELU6_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReLU6 = "ReLU6";
/// \brief Computes ReLU (Rectified Linear Unit) upper bounded by 6 of input tensors element-wise.
/// Refer to Python API @ref mindspore.ops.ReLU6 for more details.
class MIND_API ReLU6 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ReLU6);
  /// \brief Constructor.
  ReLU6() : BaseOperator(kNameReLU6) { InitIOName({"x"}, {"output"}); }
  /// \brief Init.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_RELU6_H_
