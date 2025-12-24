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

#ifndef MINDSPORE_CORE_OPS_DYNAMIC_GRU_V2_H_
#define MINDSPORE_CORE_OPS_DYNAMIC_GRU_V2_H_
#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDynamicGRUV2 = "DynamicGRUV2";
class OPS_API DynamicGRUV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DynamicGRUV2);

  /// \brief Constructor.
  DynamicGRUV2() : BaseOperator(kNameDynamicGRUV2) {
    InitIOName({"x", "weight_input", "weight_hidden", "bias_input", "bias_hidden", "seq_length", "init_h"},
               {"y", "output_h", "update", "reset", "new", "hidden_new"});
  }
};
AbstractBasePtr DynamicGRUV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args);

using PrimDynamicGRUV2Ptr = std::shared_ptr<DynamicGRUV2>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_DYNAMIC_GRU_V2_H_
