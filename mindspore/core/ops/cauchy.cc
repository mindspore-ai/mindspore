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
#include "ops/cauchy.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
abstract::ShapePtr CauchyInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", input_args.size(), kGreaterEqual, 0, prim_name);
  MS_EXCEPTION_IF_NULL(primitive->GetAttr("size"));
  auto size = GetValue<std::vector<int64_t>>(primitive->GetAttr("size"));
  (void)CheckAndConvertUtils::CheckInteger("the length of 'size'", size.size(), kGreaterThan, 0, prim_name);
  return std::make_shared<abstract::Shape>(size);
}

MIND_API_OPERATOR_IMPL(Cauchy, BaseOperator);

abstract::AbstractBasePtr CauchyInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);

  auto infer_shape = CauchyInferShape(primitive, input_args);
  auto infer_type = std::make_shared<TensorType>(kFloat32);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Cauchy, prim::kPrimCauchy, CauchyInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
