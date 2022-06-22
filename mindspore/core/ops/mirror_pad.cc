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
#include "ops/mirror_pad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kInputSize = 2;
constexpr int64_t kNumTwo = 2;
}  // namespace
void MirrorPad::Init(const std::string &pad_mode) { this->set_mode(pad_mode); }

void MirrorPad::set_mode(const std::string &mode) { (void)this->AddAttr(kNamePadMode, api::MakeValue(mode)); }

std::string MirrorPad::get_mode() const { return GetValue<std::string>(GetAttr(kNamePadMode)); }

MIND_API_OPERATOR_IMPL(MirrorPad, BaseOperator);

namespace {
TypePtr MirrorPadInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,   kUInt16,
                                         kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kBool};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, name);
}

abstract::ShapePtr MirrorPadInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
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
  if (paddings_size % kNumTwo == 1) {
    MS_EXCEPTION(ValueError) << "For 'mirror pad', the length of 'paddings' should be even, but got " << paddings_size;
  }
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputSize,
                                           prim_name);

  auto input_x_shape_ptr = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
  if (input_x_shape_ptr->IsDynamic()) {
    return input_args[0]->BuildShape()->cast<abstract::ShapePtr>();
  }
  for (size_t i = 0; i < static_cast<size_t>(size); i++) {
    x_shape[i] = x_shape[i] + paddings_arg[kNumTwo * i] + paddings_arg[kNumTwo * i + 1];
  }
  return std::make_shared<abstract::Shape>(x_shape);
}
}  // namespace

abstract::AbstractBasePtr MirrorPadInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infertype = MirrorPadInferType(primitive, input_args);
  auto infershape = MirrorPadInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MirrorPad, prim::kPrimMirrorPad, MirrorPadInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
