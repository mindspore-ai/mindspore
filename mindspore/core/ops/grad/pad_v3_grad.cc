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
#include "ops/grad/pad_v3_grad.h"
#include <set>
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PadV3GradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  constexpr size_t kTwo = 2;
  constexpr size_t kThree = 3;
  constexpr size_t kFour = 4;
  constexpr size_t kFive = 5;
  constexpr size_t kPaddingsSizeTwo = 2;
  constexpr size_t kPaddingsSizeFour = 4;
  constexpr size_t kPaddingsSizeSix = 6;
  constexpr size_t paddings_pos_2 = 2;
  constexpr size_t paddings_pos_3 = 3;
  constexpr size_t paddings_pos_4 = 4;
  constexpr size_t paddings_pos_5 = 5;
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
  int64_t paddings_size = SizeToLong(paddings_arg.size());
  std::vector<int64_t> paddings_val;
  for (int64_t i = 0; i < paddings_size; ++i) {
    paddings_val.push_back(int64_t(paddings_arg[LongToSize(i)]));
  }
  auto paddings_contiguous = GetValue<bool>(primitive->GetAttr("paddings_contiguous"));
  if (paddings_contiguous == false) {
    std::vector<int64_t> tmp = paddings_val;
    for (int64_t i = 0; i < paddings_size; ++i) {
      if (i % SizeToLong(kTwo) == 0) {
        paddings_val[LongToSize(i)] = tmp[LongToSize(i) / kTwo];
      } else {
        paddings_val[LongToSize(i)] = tmp[LongToSize(i + paddings_size) / kTwo];
      }
    }
  }
  primitive->set_attr("padding_switched", MakeValue(paddings_val));
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  std::vector<int64_t> out_shape;
  if (paddings_size == kPaddingsSizeTwo) {
    (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 2", kThree, kEqual,
                                             SizeToLong(x_shape.size()), prim_name);
    (void)out_shape.emplace_back(x_shape[0]);
    (void)out_shape.emplace_back(x_shape[1]);
    (void)out_shape.emplace_back(x_shape[kInputIndex2] - paddings_val[0] - paddings_val[1]);
  } else if (paddings_size == kPaddingsSizeFour) {
    (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 4", kFour, kEqual,
                                             SizeToLong(x_shape.size()), prim_name);
    (void)out_shape.emplace_back(x_shape[0]);
    (void)out_shape.emplace_back(x_shape[1]);
    (void)out_shape.emplace_back(x_shape[kInputIndex2] - paddings_val[paddings_pos_2] - paddings_val[paddings_pos_3]);
    (void)out_shape.emplace_back(x_shape[kInputIndex3] - paddings_val[0] - paddings_val[1]);
  } else if (paddings_size == kPaddingsSizeSix) {
    (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 6", kFive, kEqual,
                                             SizeToLong(x_shape.size()), prim_name);
    (void)out_shape.emplace_back(x_shape[0]);
    (void)out_shape.emplace_back(x_shape[1]);
    (void)out_shape.emplace_back(x_shape[kInputIndex2] - paddings_val[paddings_pos_4] - paddings_val[paddings_pos_5]);
    (void)out_shape.emplace_back(x_shape[kInputIndex3] - paddings_val[paddings_pos_2] - paddings_val[paddings_pos_3]);
    (void)out_shape.emplace_back(x_shape[kInputIndex4] - paddings_val[0] - paddings_val[1]);
  } else {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the length of paddings must be 2, 4 or 6, but got "
                             << paddings_size;
  }
  (void)CheckAndConvertUtils::CheckPositiveVector("out_shape", out_shape, prim_name);
  auto x_shape_ptr = input_args[0]->isa<abstract::AbstractTensor>()
                       ? input_args[0]->cast<abstract::AbstractTensorPtr>()->BuildShape()
                       : input_args[0]->cast<abstract::AbstractTuplePtr>()->BuildShape();
  if (!x_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::Shape>(out_shape);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr PadV3GradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> args = {{"x", input_args[0]->BuildType()}};
  return CheckAndConvertUtils::CheckTensorTypeSame(args,
                                                   {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64,
                                                    kFloat16, kFloat32, kFloat64, kComplex64, kComplex128},
                                                   prim->name());
}
}  // namespace

AbstractBasePtr PadV3GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = PadV3GradInferType(primitive, input_args);
  auto infer_shape = PadV3GradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

bool PadV3Grad::get_paddings_contiguous() const { return GetValue<bool>(GetAttr("paddings_contiguous")); }
std::string PadV3Grad::get_mode() const { return GetValue<string>(GetAttr("mode")); }
std::vector<int64_t> PadV3Grad::get_paddings() const {
  return GetValue<std::vector<int64_t>>(GetAttr("padding_switched"));
}

MIND_API_OPERATOR_NAME_IMPL(PadV3Grad, kNamePadV3Grad, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(PadV3Grad, prim::kPrimPadV3Grad, PadV3GradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
