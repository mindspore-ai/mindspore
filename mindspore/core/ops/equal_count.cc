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

#include "ops/equal_count.h"
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <complex>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr EqualCountInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();

  auto input0 = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
  auto input1 = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 1);
  abstract::CheckShapeSame(op_name, input0, input1);

  std::vector<int64_t> out_shape = {1};
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr EqualCountInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim->name(), input_args, 0);
  auto y = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim->name(), input_args, 1);
  (void)abstract::CheckDtypeSame(prim->name(), x, y);

  return input_args[0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(EqualCount, BaseOperator);
AbstractBasePtr EqualCountInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  auto type = EqualCountInferType(primitive, input_args);
  auto shape = EqualCountInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(EqualCount, prim::kPrimEqualCount, EqualCountInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
