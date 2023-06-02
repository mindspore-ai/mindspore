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

#ifndef MINDSPORE_CORE_OPS_SOFTMAX_H_
#define MINDSPORE_CORE_OPS_SOFTMAX_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSoftmax = "Softmax";
/// \brief Softmax operation. Refer to Python API @ref mindspore.ops.Softmax for more details.
class MIND_API Softmax : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Softmax);
  /// \brief Constructor.
  Softmax() : BaseOperator(kNameSoftmax) { InitIOName({"x"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Softmax for the inputs.
  void Init(const int64_t axis = -1);
  /// \brief Set axis.
  void set_axis(const std::vector<int64_t> &axis);
  /// \brief Get axis.
  ///
  /// \return axis.
  std::vector<int64_t> get_axis() const;
};

MIND_API abstract::AbstractBasePtr SoftmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SOFTMAX_H_
