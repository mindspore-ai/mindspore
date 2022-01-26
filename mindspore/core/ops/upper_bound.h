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

#ifndef MINDSPORE_CORE_OPS_UPPER_BOUND_H_
#define MINDSPORE_CORE_OPS_UPPER_BOUND_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/primitive_infer_map.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameUpperBound = "UpperBound";
class UpperBound : public PrimitiveC {
 public:
  UpperBound() : PrimitiveC(kNameUpperBound) { InitIOName({"sorted_x", "values"}, {"y"}); }
  ~UpperBound() = default;
  MS_DECLARE_PARENT(UpperBound, PrimitiveC);
};
AbstractBasePtr UpperBoundInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args);
using PrimUpperBound = std::shared_ptr<UpperBound>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_UPPER_BOUND_H_
