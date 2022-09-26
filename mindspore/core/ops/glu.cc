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
#include "ops/glu.h"

#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(GLU, BaseOperator);
void GLU::Init(int64_t axis) { set_axis(axis); }

void GLU::set_axis(int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }

int64_t GLU::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}
namespace {

abstract::ShapePtr GLUInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kEvenNum = 2;
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("rank of x", x_rank, kGreaterEqual, 1, prim_name);
  auto axis = GetValue<int64_t>(primitive->GetAttr("axis"));
  CheckAndConvertUtils::CheckInRange("axis", axis, kIncludeLeft, {-x_rank, x_rank}, prim_name);
  if (axis < 0) {
    axis += x_rank;
  }
  auto shape = x_shape;
  auto shape_of_split_dim = x_shape[LongToSize(axis)];
  if (shape_of_split_dim % kEvenNum != 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', x.shape[" << axis << "] must be even, but got "
                             << shape_of_split_dim << " .";
  }
  shape[axis] = shape_of_split_dim / kEvenNum;
  return std::make_shared<abstract::Shape>(shape);
}

TypePtr GLUInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
  return input_args[0]->BuildType();
}
}  // namespace
AbstractBasePtr GLUInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputsNum, primitive->name());
  auto infer_type = GLUInferType(primitive, input_args);
  auto infer_shape = GLUInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
//
REGISTER_PRIMITIVE_EVAL_IMPL(GLU, prim::kPrimGLU, GLUInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
