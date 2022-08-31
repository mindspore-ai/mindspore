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

#include <set>
#include <utility>
#include "ops/grad/mirror_pad_grad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kPaddingsSecondDim = 2;
constexpr int64_t kMaxPaddings = 5;
abstract::ShapePtr MirrorPadGradInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto paddings = input_args[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(paddings);
  auto paddings_arg = CheckAndConvertUtils::CheckTensorIntValue("paddings", paddings, prim_name);
  std::vector<std::pair<int64_t, int64_t>> paddings_attr;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto paddings_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto mode = GetValue<std::string>(primitive->GetAttr("mode"));
  if (paddings_shape.size() != kPaddingsSecondDim) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', paddings must be equal to 2 dims, but got "
                             << paddings_shape.size();
  }
  if (paddings_shape[1] != kPaddingsSecondDim) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', paddings must be a matrix with 2 columns, but got "
                             << paddings_shape[1];
  }
  if (static_cast<size_t>(paddings_shape[0]) != x_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', paddings.shape[0] must equal to input's rank, but got "
                             << paddings_shape[0];
  }
  for (size_t i = 0; i < paddings_arg.size(); i = i + kPaddingsSecondDim) {
    paddings_attr.push_back(std::make_pair(paddings_arg[i], paddings_arg[i + 1]));
  }
  (void)CheckAndConvertUtils::CheckInteger("paddings_size", paddings_attr.size(), kEqual, x_shape.size(), prim_name);
  int64_t size = x_shape.size();
  if (size < 0 || size > kMaxPaddings) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the dimension of input only supports less than or equal to 5 dims, but got " << size
                             << " dims";
  }
  auto input_x_shape_ptr = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
  if (input_x_shape_ptr->IsDynamic()) {
    return input_args[0]->BuildShape()->cast<abstract::ShapePtr>();
  }
  for (int64_t i = 0; i < size; i++) {
    if (paddings_attr[i].first < 0 || paddings_attr[i].second < 0) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', all elements of paddings must be >= 0.";
    }
    int64_t out_size = x_shape[i] - (paddings_attr[i].first + paddings_attr[i].second);
    if (mode == "SYMMETRIC") {
      if (paddings_attr[i].first > out_size || paddings_attr[i].second > out_size)
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', paddings must be no greater "
                                    "than the dimension size: ["
                                 << paddings_attr[i].first << "], [" << paddings_attr[i].second << "] greater than ["
                                 << out_size << "]";
    } else if (mode == "REFLECT") {
      if (paddings_attr[i].first >= out_size || paddings_attr[i].second >= out_size)
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', paddings must be no greater "
                                    "than the dimension size: ["
                                 << paddings_attr[i].first << "], [" << paddings_attr[i].second << "] not less than ["
                                 << out_size << "]";
    }
  }
  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < x_shape.size(); i++) {
    (void)out_shape.emplace_back(x_shape[i] - paddings_attr[i].first - paddings_attr[i].second);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr MirrorPadGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  (void)CheckAndConvertUtils::CheckTensorTypeValid("paddings", input_args[1]->BuildType(), {kInt32, kInt64},
                                                   prim->name());
  return CheckAndConvertUtils::CheckTensorTypeValid(
    "input_x", input_args[0]->BuildType(),
    {kInt8, kInt16, kInt32, kInt64, kUInt, kUInt8, kUInt16, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128},
    prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(MirrorPadGrad, BaseOperator);
AbstractBasePtr MirrorPadGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = MirrorPadGradInferType(primitive, input_args);
  auto infer_shape = MirrorPadGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MirrorPadGrad, prim::kPrimMirrorPadGrad, MirrorPadGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
