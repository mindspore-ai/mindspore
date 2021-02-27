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

#include "ops/fake_quant_with_min_max_vars_per_channel.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void FakeQuantWithMinMaxVarsPerChannel::Init(const int64_t num_bits, const bool narrow_range) {
  this->set_num_bits(num_bits);
  this->set_narrow_range(narrow_range);
}
void FakeQuantWithMinMaxVarsPerChannel::set_num_bits(const int64_t num_bits) {
  CheckAndConvertUtils::CheckInteger(kNumBits, num_bits, kGreaterThan, 0, this->name());
  this->AddAttr(kNumBits, MakeValue(num_bits));
}
void FakeQuantWithMinMaxVarsPerChannel::set_narrow_range(const bool narrow_range) {
  this->AddAttr(kNarrowRange, MakeValue(narrow_range));
}
int64_t FakeQuantWithMinMaxVarsPerChannel::get_num_bits() const {
  auto value_ptr = GetAttr(kNumBits);
  return GetValue<int64_t>(value_ptr);
}
bool FakeQuantWithMinMaxVarsPerChannel::get_narrow_range() const {
  auto value_ptr = GetAttr(kNarrowRange);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr FakeQuantWithMinMaxVarsPerChannelInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto FakeQuantWithMinMaxVarsPerChannel_prim = primitive->cast<PrimFakeQuantWithMinMaxVarsPerChannelPtr>();
  MS_EXCEPTION_IF_NULL(FakeQuantWithMinMaxVarsPerChannel_prim);
  auto op_name = FakeQuantWithMinMaxVarsPerChannel_prim->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), op_name);
  auto min_shape = CheckAndConvertUtils::ConvertShapePtrToShape("min_shape", input_args[1]->BuildShape(), op_name);
  auto max_shape = CheckAndConvertUtils::ConvertShapePtrToShape("max_shape", input_args[2]->BuildShape(), op_name);
  CheckAndConvertUtils::CheckInteger("x rank", (int64_t)x_shape.size(), kGreaterThan, 1, op_name);
  CheckAndConvertUtils::Check("min shape", min_shape, kEqual, "max shape", max_shape, op_name);
  CheckAndConvertUtils::CheckInteger("min shape", (int64_t)min_shape.size(), kEqual, 1, op_name);
  CheckAndConvertUtils::Check("min shape", min_shape[0], kEqual, "x shape", x_shape[x_shape.size() - 1], op_name);

  auto x_type = input_args[0]->BuildType();
  auto min_type = input_args[1]->BuildType();
  auto max_type = input_args[2]->BuildType();
  std::vector<std::string> type_name = {"x", "min", "max"};
  std::vector<TypePtr> type = {x_type, min_type, max_type};
  for (int64_t i = 0; i < 3; i++) {
    CheckAndConvertUtils::CheckTensorTypeValid(type_name[i], type[i], {kNumberTypeFloat16, kNumberTypeFloat32},
                                               op_name);
  }
  auto tensor_type = x_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  return std::make_shared<abstract::AbstractTensor>(data_type, x_shape);
}
REGISTER_PRIMITIVE_C(kNameFakeQuantWithMinMaxVarsPerChannel, FakeQuantWithMinMaxVarsPerChannel);
}  // namespace ops
}  // namespace mindspore
