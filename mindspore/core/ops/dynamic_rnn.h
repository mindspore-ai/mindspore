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

#ifndef MINDSPORE_CORE_OPS_DYNAMIC_RNN_H_
#define MINDSPORE_CORE_OPS_DYNAMIC_RNN_H_
#include <string>
#include <vector>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDynamicRNN = "DynamicRNN";
class MIND_API DynamicRNN : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DynamicRNN);

  DynamicRNN() : BaseOperator(kNameDynamicRNN) {
    InitIOName({"x", "w", "b", "seq_length", "init_h", "init_c"},
               {"y", "output_h", "output_c", "i", "j", "f", "o", "tanhc"});
  }
  void Init() const {}
};

MIND_API abstract::AbstractBasePtr DynamicRNNInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DYNAMIC_RNN_H_
