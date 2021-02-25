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

#include "ops/bias_add.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
// Add
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
// Add
namespace {
abstract::ShapePtr BiasAddInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // check
  CheckAndConvertUtils::CheckInteger("biasadd_infer", input_args.size(), kEqual, 2, prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto b_shape = CheckAndConvertUtils::ConvertShapePtrToShape("b_shape", input_args[1]->BuildShape(), prim_name);
  CheckAndConvertUtils::CheckInteger("x rank", x_shape.size(), kGreaterEqual, 2, prim_name);
  CheckAndConvertUtils::CheckInteger("bias rank", b_shape.size(), kEqual, 1, prim_name);
  auto format = Format(GetValue<int64_t>(primitive->GetAttr(kFormat)));
  auto x_channel = x_shape[1];
  if (format != NCHW) {
    x_channel = x_shape[x_shape.size() - 1];
  }
  CheckAndConvertUtils::Check("b_shape[0]", b_shape[0], kEqual, "x_shape[1]", x_channel, prim_name);

  std::vector<int64_t> out_shape = x_shape;
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr BiasAddInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  CheckAndConvertUtils::CheckInteger("biasadd_infer", input_args.size(), kEqual, 2, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  types.emplace("input_x", input_args[0]->BuildType());
  types.emplace("bias", input_args[1]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
  return TypeIdToType(infer_type);
}
}  // namespace
void BiasAdd::set_format(const Format &format) {
  int64_t f = format;
  this->AddAttr(kFormat, MakeValue(f));
}
Format BiasAdd::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}
void BiasAdd::Init(const Format &format) { this->set_format(format); }
AbstractBasePtr BiasAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(BiasAddInferType(primitive, input_args),
                                                    BiasAddInferShape(primitive, input_args));
}
// Add
REGISTER_PRIMITIVE_C(kNameBiasAdd, BiasAdd);
}  // namespace ops
}  // namespace mindspore
