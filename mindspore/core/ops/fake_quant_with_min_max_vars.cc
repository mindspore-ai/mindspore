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
#include <map>
#include <string>
#include <vector>
#include "ops/fake_quant_with_min_max_vars.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto fake_prim = primitive->cast<PrimFakeQuantWithMinMaxVarsPtr>();
  MS_EXCEPTION_IF_NULL(fake_prim);
  auto prim_name = fake_prim->name();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto min_shape = CheckAndConvertUtils::ConvertShapePtrToShape("min_shape", input_args[1]->BuildShape(), prim_name);
  auto max_shape = CheckAndConvertUtils::ConvertShapePtrToShape("max_shape", input_args[2]->BuildShape(), prim_name);
  CheckAndConvertUtils::CheckInteger("x_rank", in_shape.size(), kGreaterEqual, 1, prim_name);
  CheckAndConvertUtils::Check("min_shape", min_shape, kEqual, "max_shape", max_shape, prim_name);
  CheckAndConvertUtils::CheckInteger("min_shape", min_shape.size(), kEqual, 1, prim_name);
  int64_t shape_val = 1;
  for (size_t i = 0; i < in_shape.size(); i++) {
    shape_val = shape_val * in_shape[i];
    if (min_shape[0] > 1 && min_shape[0] != shape_val) {
      MS_EXCEPTION(ValueError) << "For" + prim_name + " the shape of \'min\' cannot broadcast to the shape of  \'x\'";
    }
  }
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  if (std::any_of(input_args.begin(), input_args.end(), [](AbstractBasePtr a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  std::map<std::string, TypePtr> types;
  types.emplace("x", input_args[0]->BuildType());
  types.emplace("min", input_args[1]->BuildType());
  types.emplace("max", input_args[2]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return TypeIdToType(infer_type);
}
}  // namespace
void FakeQuantWithMinMaxVars::Init(const bool narrow_range, const int64_t num_bits) {
  this->set_narrow_range(narrow_range);
  this->set_num_bits(num_bits);
}

void FakeQuantWithMinMaxVars::set_narrow_range(const bool narrow_range) {
  this->AddAttr(kNarrowRange, MakeValue(narrow_range));
}

bool FakeQuantWithMinMaxVars::get_narrow_range() const {
  auto value_ptr = this->GetAttr(kNarrowRange);
  return GetValue<bool>(value_ptr);
}

void FakeQuantWithMinMaxVars::set_num_bits(const int64_t num_bits) { this->AddAttr(kNumBits, MakeValue(num_bits)); }

int64_t FakeQuantWithMinMaxVars::get_num_bits() const {
  auto value_ptr = this->GetAttr(kNumBits);
  return GetValue<int64_t>(value_ptr);
}
AbstractBasePtr FakeQuantWithMinMaxVarsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameFakeQuantWithMinMaxVars, FakeQuantWithMinMaxVars);
}  // namespace ops
}  // namespace mindspore
