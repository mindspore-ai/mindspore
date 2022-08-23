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
#include "ops/pad_v3.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kConstantMaxDims = 5;
constexpr int64_t kReflectMaxDims = 4;
constexpr int64_t nTwo = 2;
constexpr int64_t nFour = 4;
constexpr int64_t nSix = 6;
constexpr int64_t padding_pos_1 = 1;
constexpr int64_t padding_pos_2 = 2;
constexpr int64_t padding_pos_3 = 3;
constexpr int64_t padding_pos_4 = 4;
abstract::ShapePtr PadV3InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto paddings = input_args[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(paddings);
  std::vector<int64_t> paddings_arg;
  if (paddings->isa<tensor::Tensor>()) {
    paddings_arg = CheckAndConvertUtils::CheckTensorIntValue("paddings value", paddings, prim_name);
  } else {
    paddings_arg = CheckAndConvertUtils::CheckTupleInt("paddings tuple value", paddings, prim_name);
  }
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  int64_t size = SizeToLong(x_shape.size());
  int64_t paddings_size = SizeToLong(paddings_arg.size());
  if (paddings_size % nTwo == 1) {
    MS_EXCEPTION(ValueError) << "For 'PadV3', the length of 'paddings' should be even, but got " << paddings_size;
  }
  int64_t paddings_dim = paddings_size / 2;
  (void)CheckAndConvertUtils::CheckInteger("length of padding", paddings_size, kLessEqual, nSix, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("pad dims", paddings_dim, kLessEqual, size, prim_name);
  auto mode = GetValue<string>(primitive->GetAttr("mode"));
  if (mode != kReflect) {
    (void)CheckAndConvertUtils::CheckInteger("input dims for constant or edge mode", size, kLessEqual, kConstantMaxDims,
                                             prim_name);
  } else {
    (void)CheckAndConvertUtils::CheckInteger("input dims for reflect mode", size, kLessEqual, kReflectMaxDims,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("length of padding for reflect mode", paddings_size, kLessEqual, nFour,
                                             prim_name);
  }

  std::vector<int64_t> paddings_val;
  for (int64_t i = 0; i < paddings_size; ++i) {
    paddings_val.push_back(int64_t(paddings_arg[LongToSize(i)]));
  }
  auto paddings_contiguous = GetValue<bool>(primitive->GetAttr("paddings_contiguous"));
  if (paddings_contiguous == false) {
    std::vector<int64_t> tmp = paddings_val;
    switch (paddings_size) {
      case nTwo:
        break;
      case nFour:
        paddings_val[padding_pos_1] = tmp[padding_pos_2];
        paddings_val[padding_pos_2] = tmp[padding_pos_1];
        break;
      case nSix:
        paddings_val[padding_pos_1] = tmp[padding_pos_3];
        paddings_val[padding_pos_2] = tmp[padding_pos_1];
        paddings_val[padding_pos_3] = tmp[padding_pos_4];
        paddings_val[padding_pos_4] = tmp[padding_pos_2];
        break;
      default:
        break;
    }
  }
  primitive->set_attr("padding_switched", MakeValue(paddings_val));
  std::vector<std::pair<int64_t, int64_t>> paddings_attr;
  for (int64_t i = 0; i < size; ++i) {
    if (nTwo * i >= paddings_size) {
      paddings_attr.push_back(std::make_pair(int64_t(0), int64_t(0)));
    } else {
      paddings_attr.push_back(
        std::make_pair(paddings_val[LongToSize(nTwo * i)], paddings_val[LongToSize(nTwo * i + 1)]));
    }
  }
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < size; ++i) {
    int64_t now_dim_size = x_shape[LongToSize(i)] + paddings_attr[LongToSize(size - i - 1)].first +
                           paddings_attr[LongToSize(size - i - 1)].second;
    if (now_dim_size < 0) {
      (void)CheckAndConvertUtils::CheckInteger("output size", now_dim_size, kGreaterEqual, 0, prim_name);
    }
    (void)out_shape.emplace_back(now_dim_size);
  }

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr PadV3InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> padding = {{"padding", input_args[1]->BuildType()}};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(padding, {kInt32, kInt64}, prim->name());

  std::map<std::string, TypePtr> args = {{"x", input_args[0]->BuildType()}};
  return CheckAndConvertUtils::CheckTensorTypeSame(args,
                                                   {kInt, kInt8, kInt16, kInt32, kInt64, kUInt, kUInt8, kUInt16, kFloat,
                                                    kFloat16, kFloat32, kFloat64, kComplex64, kComplex128},
                                                   prim->name());
}
}  // namespace

AbstractBasePtr PadV3Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t kConstantInput = 3;
  constexpr int64_t kOtherInput = 2;
  auto mode = GetValue<string>(primitive->GetAttr("mode"));
  if (mode == kConstant) {
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kConstantInput, primitive->name());
  } else {
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kOtherInput, primitive->name());
  }

  auto infer_type = PadV3InferType(primitive, input_args);
  auto infer_shape = PadV3InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

bool PadV3::get_paddings_contiguous() const { return GetValue<bool>(GetAttr("paddings_contiguous")); }
std::string PadV3::get_mode() const { return GetValue<string>(GetAttr("mode")); }
std::vector<int64_t> PadV3::get_paddings() const { return GetValue<std::vector<int64_t>>(GetAttr("padding_switched")); }

MIND_API_OPERATOR_NAME_IMPL(PadV3, kNamePadV3, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(PadV3, prim::kPrimPadV3, PadV3Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
