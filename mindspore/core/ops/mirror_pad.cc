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
#include <string>
#include "ops/mirror_pad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kPaddingsSecondDimSize = 2;
constexpr int64_t MAX_PADDINGS = 5;
constexpr int64_t kMirrorPadInputNum = 2;
}  // namespace

void MirrorPad::set_mode(const std::string &mode) { (void)AddAttr(kMode, api::MakeValue(mode)); }
std::string MirrorPad::get_mode() const { return GetValue<std::string>(GetAttr(kMode)); }

void CheckPaddingParam(const std::vector<int64_t> &paddings_shape, const std::vector<int64_t> &x_shape,
                       const std::string &prim_name) {
  if (paddings_shape.size() != kPaddingsSecondDimSize) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', paddings must be equal to 2 dims, but got "
                             << paddings_shape.size();
  }
  if (paddings_shape[1] != kPaddingsSecondDimSize) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', paddings must be a matrix with 2 columns, but got "
                             << paddings_shape[1];
  }
  if (static_cast<size_t>(paddings_shape[0]) != x_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', paddings.shape[0] must equal to input's rank, but got "
                             << paddings_shape[0];
  }
  MS_LOG(DEBUG) << "For '" << prim_name << "' padding shape:  " << paddings_shape;
  return;
}

void CheckPaddingValue(const std::vector<std::pair<int64_t, int64_t>> &paddings_attr,
                       const std::vector<int64_t> &x_shape, const std::string &mode, const std::string &prim_name) {
  int64_t size = static_cast<int64_t>(x_shape.size());
  if (size < 0 || size > MAX_PADDINGS) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the dimension of input only supports less than or equal to 5 dims, but got " << size
                             << " dims";
  }
  for (int64_t i = 0; i < size; i++) {
    if (x_shape[i] == abstract::Shape::kShapeDimAny) {
      continue;
    }
    if (paddings_attr[i].first < 0 || paddings_attr[i].second < 0) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', all elements of paddings must be >= 0.";
    }
    if (mode == "SYMMETRIC") {
      if (paddings_attr[i].first > static_cast<int64_t>(x_shape[i]) ||
          paddings_attr[i].second > static_cast<int64_t>(x_shape[i])) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', paddings must be no greater "
                                    "than the dimension size: ["
                                 << paddings_attr[i].first << "], [" << paddings_attr[i].second << "] greater than ["
                                 << static_cast<int64_t>(x_shape[i]) << "]";
      }
    } else if (mode == "REFLECT") {
      if (paddings_attr[i].first >= static_cast<int64_t>(x_shape[i]) ||
          paddings_attr[i].second >= static_cast<int64_t>(x_shape[i])) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', paddings must be no greater "
                                    "than the dimension size: ["
                                 << paddings_attr[i].first << "], [" << paddings_attr[i].second << "] not less than ["
                                 << static_cast<int64_t>(x_shape[i]) << "]";
      }
    }
  }
}

MIND_API_OPERATOR_IMPL(MirrorPad, BaseOperator);
class MirrorPadInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kMirrorPadInputNum, primitive->name());
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto input_x_shape_ptr = input_args[0]->BuildShape();
    MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
    auto input_x_shape = input_x_shape_ptr->cast<abstract::ShapePtr>();
    // Dynamic rank process.
    if (IsDynamicRank(input_x_shape->shape())) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }
    auto paddings = input_args[1]->BuildValue();
    MS_EXCEPTION_IF_NULL(paddings);
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
    auto paddings_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
    CheckPaddingParam(paddings_shape, x_shape, prim_name);
    // if shape of x is determined and padding value is unknown, return a all -1 shape
    if (paddings->isa<AnyValue>() || paddings->isa<None>()) {
      return std::make_shared<abstract::Shape>(ShapeVector(x_shape.size(), abstract::Shape::kShapeDimAny));
    }
    auto paddings_arg = CheckAndConvertUtils::CheckTensorIntValue(kPaddings, paddings, prim_name);
    std::vector<std::pair<int64_t, int64_t>> paddings_attr;

    auto mode = GetValue<std::string>(primitive->GetAttr(kMode));
    for (size_t i = 0; i < paddings_arg.size(); i = i + static_cast<size_t>(kPaddingsSecondDimSize)) {
      paddings_attr.push_back(std::make_pair(paddings_arg[i], paddings_arg[i + 1]));
    }
    (void)CheckAndConvertUtils::CheckInteger(kPaddingsSize, SizeToLong(paddings_attr.size()), kEqual,
                                             SizeToLong(x_shape.size()), prim_name);
    CheckPaddingValue(paddings_attr, x_shape, mode, prim_name);
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < x_shape.size(); i++) {
      // In dynamic situation , if input axis is dynamic, output axis is dynamic too.
      if (x_shape[i] == abstract::Shape::kShapeDimAny) {
        (void)out_shape.emplace_back(abstract::Shape::kShapeDimAny);
      } else {
        (void)out_shape.emplace_back(x_shape[i] + paddings_attr[i].first + paddings_attr[i].second);
      }
    }
    return std::make_shared<abstract::Shape>(out_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kMirrorPadInputNum, prim->name());
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    (void)CheckAndConvertUtils::CheckTensorTypeValid("paddings", input_args[1]->BuildType(), {kInt32, kInt64},
                                                     prim->name());
    return CheckAndConvertUtils::CheckTensorTypeValid(
      "input_x", input_args[0]->BuildType(),
      {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool},
      prim->name());
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MirrorPad, prim::kPrimMirrorPad, MirrorPadInfer, false);
}  // namespace ops
}  // namespace mindspore
