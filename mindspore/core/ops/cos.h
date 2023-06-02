/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_COS_H_
#define MINDSPORE_CORE_OPS_COS_H_
#include <memory>
#include <vector>

#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCos = "Cos";

/// \brief Computes cosine of input element-wise. Refer to Python API @ref mindspore.ops.Cos for more details.
class MIND_API Cos : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Cos);
  /// \brief Constructor.
  Cos() : BaseOperator(kNameCos) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Cos for the inputs.
  void Init(float alpha = 0.0);
};

MIND_API abstract::AbstractBasePtr CosInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_COS_H_
