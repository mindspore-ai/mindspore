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

#ifndef MINDSPORE_CORE_OPS_BN_INFER_H_
#define MINDSPORE_CORE_OPS_BN_INFER_H_
#include <vector>
#include <memory>
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBNInfer = "ABNInfer";
class MIND_API BNInfer : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(BNInfer);
  /// \brief Constructor.
  BNInfer() : BaseOperator(kNameBNInfer) {
    InitIOName({"x", "scale", "offset", "mean", "variance"},
               {"y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"});
  }
};

MIND_API abstract::AbstractBasePtr BNInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BN_INFER_H_
