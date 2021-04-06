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
void LogSoftmax::set_axis(const int64_t axis) { this->AddAttr(kAxis, MakeValue(axis)); }

int64_t LogSoftmax::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

void LogSoftmax::Init(const int64_t axis) { this->set_axis(axis); }

abstract::ShapePtr LogSoftmaxInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto LogSoftmax_prim = primitive->cast<PrimLogSoftmaxPtr>();
  MS_EXCEPTION_IF_NULL(LogSoftmax_prim);
  auto op_name = LogSoftmax_prim->name();
  auto axis = LogSoftmax_prim->get_axis();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->GetShapeTrack(), op_name);
  auto rank = SizeToLong(in_shape.size());
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeLeft, {-rank, rank}, op_name);
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr LogSoftmaxInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
}

AbstractBasePtr LogSoftmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(LogSoftmaxInferType(primitive, input_args),
                                                    LogSoftmaxInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameLogSoftmax, LogSoftmax);
}  // namespace ops
}  // namespace mindspore
