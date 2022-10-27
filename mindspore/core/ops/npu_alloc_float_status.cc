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

#include <map>
#include <string>

#include "ops/npu_alloc_float_status.h"
#include "ops/op_utils.h"
#include "abstract/param_validator.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kFloatStatusNum = 8;
}
MIND_API_OPERATOR_IMPL(NPUAllocFloatStatus, BaseOperator);
AbstractBasePtr NPUAllocFloatStatusInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  ShapeVector output_shape;
  output_shape.push_back(kFloatStatusNum);
  return abstract::MakeAbstract(std::make_shared<abstract::Shape>(output_shape), kTensorTypeFP32);
}
REGISTER_PRIMITIVE_EVAL_IMPL(NPUAllocFloatStatus, prim::kPrimNPUAllocFloatStatus, NPUAllocFloatStatusInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
