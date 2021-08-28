/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/log_softmax.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void LogSoftmax::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, MakeValue(axis)); }

int64_t LogSoftmax::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

void LogSoftmax::Init(const int64_t axis) { this->set_axis(axis); }

abstract::ShapePtr LogSoftmaxInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto axis = GetValue<int64_t>(primitive->GetAttr(kAxis));
  (void)CheckAndConvertUtils::CheckInteger("log_softmax infer", int64_t(input_args.size()), kEqual, 1, op_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  if (shape_map.empty()) {
    // Scalar input, has no shape
    return std::make_shared<abstract::Shape>(std::vector<int64_t>());
  }
  auto in_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  auto rank = SizeToLong(in_shape.size());
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeLeft, {-rank, rank}, op_name);
  if (min_shape.size() != 0 && max_shape.size() != 0) {
    return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
  }
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr LogSoftmaxInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto op_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("log_softmax infer", int64_t(input_args.size()), kEqual, 1, op_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, op_name);
}

AbstractBasePtr LogSoftmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(LogSoftmaxInferShape(primitive, input_args),
                                LogSoftmaxInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(LogSoftmax, prim::kPrimLogSoftmax, LogSoftmaxInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
