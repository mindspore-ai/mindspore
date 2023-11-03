/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_UPSAMPLE_NEAREST1D_H_
#define MINDSPORE_CORE_OPS_UPSAMPLE_NEAREST1D_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameUpsampleNearest1d = "UpsampleNearest1d";

class MIND_API UpsampleNearest1d : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(UpsampleNearest1d);
  /// \brief Constructor.
  UpsampleNearest1d() : BaseOperator(kNameUpsampleNearest1d) {
    InitIOName({"input_tensor", "output_size", "scale_factors"}, {"output"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.UpsampleNearest1d for the inputs.
  void Init() const {}
};
MIND_API abstract::AbstractBasePtr UpsampleNearest1dInfer(const abstract::AnalysisEnginePtr &,
                                                          const PrimitivePtr &primitive,
                                                          const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_UPSAMPLE_NEAREST1D_H_
