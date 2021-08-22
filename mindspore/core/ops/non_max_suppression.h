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

#ifndef MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_H_
#define MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "abstract/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNonMaxSuppression = "NonMaxSuppression";
class NonMaxSuppression : public PrimitiveC {
 public:
  NonMaxSuppression() : PrimitiveC(kNameNonMaxSuppression) {}
  ~NonMaxSuppression() = default;
  MS_DECLARE_PARENT(NonMaxSuppression, PrimitiveC);
  void Init(const int64_t center_point_box = 0);
  void set_center_point_box(const int64_t center_point_box);
  int64_t get_center_point_box() const;
};
AbstractBasePtr NonMaxSuppressionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args);
using PrimNonMaxSuppressionPtr = std::shared_ptr<NonMaxSuppression>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_H_
