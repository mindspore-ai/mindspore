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
constexpr size_t kMaxPaddings = 5;

void verify_padding_range(const std::string &mode, int64_t out_size, const std::pair<int64_t, int64_t> padding_attr,
                          const std::string &prim_name) {
  if (padding_attr.first < 0 || padding_attr.second < 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', all elements of paddings must be >= 0.";
  }
  if (mode == "SYMMETRIC") {
    if (padding_attr.first > out_size || padding_attr.second > out_size) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', paddings must be no greater than the dimension size: ["
                               << padding_attr.first << "], [" << padding_attr.second << "] greater than [" << out_size
                               << "]";
    }
  } else if (mode == "REFLECT") {
    if (padding_attr.first >= out_size || padding_attr.second >= out_size) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', paddings must be no greater  than the dimension size: ["
                               << padding_attr.first << "], [" << padding_attr.second << "] not less than [" << out_size
                               << "]";
    }
  }
}

abstract::ShapePtr MirrorPadGradInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto input_x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
  if (input_x_shape_ptr->IsDynamic()) {  // if shape of x is dynamic, then passthrough the shape
    return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  }

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto paddings_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto paddings = input_args[kInputIndex1]->BuildValue();
  MS_EXCEPTION_IF_NULL(paddings);
  // if shape of x is determined and padding value is unknown, return a all -1 shape
  if (paddings->isa<AnyValue>() || paddings->isa<None>()) {
    return std::make_shared<abstract::Shape>(ShapeVector(x_shape.size(), UNKNOWN_DIM));
  }

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

  auto paddings_arg = CheckAndConvertUtils::CheckTensorIntValue("paddings", paddings, prim_name);
  std::vector<std::pair<int64_t, int64_t>> paddings_attr;
  for (size_t i = 0; i < paddings_arg.size(); i = i + kPaddingsSecondDim) {
    paddings_attr.push_back(std::make_pair(paddings_arg[i], paddings_arg[i + 1]));
  }
  (void)CheckAndConvertUtils::CheckInteger("paddings_size", SizeToLong(paddings_attr.size()), kEqual,
                                           SizeToLong(x_shape.size()), prim_name);
  size_t size = x_shape.size();
  if (size > kMaxPaddings) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the dimension of input only supports less than or equal to 5 dims, but got " << size
                             << " dims";
  }

  std::string mode = GetValue<std::string>(primitive->GetAttr(kMode));
  for (size_t i = 0; i < size; i++) {
    int64_t out_size = x_shape[i] - (paddings_attr[i].first + paddings_attr[i].second);
    verify_padding_range(mode, out_size, paddings_attr[i], prim_name);
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

void MirrorPadGrad::set_mode(const std::string &mode) {
  (void)CheckAndConvertUtils::CheckString(kMode, mode, {"REFLECT", "SYMMETRIC"}, this->name());
  (void)AddAttr(kMode, api::MakeValue(mode));
}

std::string MirrorPadGrad::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  return GetValue<std::string>(value_ptr);
}

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
