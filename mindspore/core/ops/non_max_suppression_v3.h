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

#ifndef MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_V3_H_
#define MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_V3_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "abstract/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNonMaxSuppressionV3 = "NonMaxSuppressionV3";
class NonMaxSuppressionV3 : public PrimitiveC {
 public:
  NonMaxSuppressionV3() : PrimitiveC(kNameNonMaxSuppressionV3) {
    InitIOName({"boxes", "score", "max_output_size", "iou_threshold", "score_threshold"}, {"selected_indices"});
  }
  ~NonMaxSuppressionV3() = default;
  MS_DECLARE_PARENT(NonMaxSuppressionV3, PrimitiveC);
};
AbstractBasePtr NonMaxSuppressionV3Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args);
using PrimNonMaxSuppressionV3Ptr = std::shared_ptr<NonMaxSuppressionV3>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_V3_H_
