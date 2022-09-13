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

#include <set>

#include "ops/op_utils.h"
#include "ops/reverse_v2.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ReverseV2InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto axis_ptr = primitive->GetAttr("axis");
  auto input_axis = GetValue<std::vector<int64_t>>(axis_ptr);
  int64_t axis_dims = input_axis.size();
  int64_t x_dims = x_shape.size();
  (void)primitive->AddAttr("axis", MakeValue(input_axis));
  const int64_t input_max_dim = 8;

  auto input_x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  if (x_dims > input_max_dim) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the dimension of input should less than 9"
                             << ", but got " << x_dims;
  }
  if (x_dims != 0) {
    std::vector<bool> reverse_shape;
    for (int64_t i = 0; i < x_dims; i++) {
      reverse_shape.push_back(false);
    }
    for (int64_t i = 0; i < axis_dims; ++i) {
      int64_t realdim = input_axis[i] < 0 ? x_dims + input_axis[i] : input_axis[i];
      input_axis[i] = realdim;
      if (realdim < 0 || realdim >= x_dims) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the 'axis[" << i << "]' must be in range of [-"
                                 << x_dims << ", " << x_dims << "), but got " << input_axis[i] << " with type 'int'.";
      } else if (realdim >= 0 && reverse_shape[realdim] == true) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << ", 'axis' cannot contain duplicate dimensions"
                                 << ", but got " << realdim;
      } else if (realdim >= 0 && reverse_shape[realdim] == false) {
        reverse_shape[realdim] = true;
      }
    }
  }
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr ReverseV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = input_args[0]->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("type", infer_type, common_valid_types_with_complex_and_bool,
                                                    prim->name());
}
}  // namespace

void ReverseV2::Init(const std::vector<int64_t> &axis) { this->set_axis(axis); }
void ReverseV2::set_axis(const std::vector<int64_t> &axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }
std::vector<int64_t> ReverseV2::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

MIND_API_OPERATOR_IMPL(ReverseV2, BaseOperator);

AbstractBasePtr ReverseV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infertype = ReverseV2InferType(primitive, input_args);
  auto infershape = ReverseV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ReverseV2, prim::kPrimReverseV2, ReverseV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
