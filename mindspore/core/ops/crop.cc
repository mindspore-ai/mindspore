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

#include <memory>
#include "ops/crop.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void Crop::Init(const int64_t axis, const std::vector<int64_t> &offsets) {
  this->set_axis(axis);
  this->set_offsets(offsets);
}

void Crop::set_axis(const int64_t axis) { this->AddAttr(kAxis, MakeValue(axis)); }

int64_t Crop::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

void Crop::set_offsets(const std::vector<int64_t> &offsets) { this->AddAttr(kOffsets, MakeValue(offsets)); }

std::vector<int64_t> Crop::get_offsets() const {
  auto value_ptr = this->GetAttr(kOffsets);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
AbstractBasePtr CropInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto crop_prim = primitive->cast<PrimCrop>();
  MS_EXCEPTION_IF_NULL(crop_prim);
  auto prim_name = crop_prim->name();
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 2, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // infer shape
  auto out_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[1]->BuildShape(), prim_name);
  // infer type
  auto x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  return std::make_shared<abstract::AbstractTensor>(x_type, out_shape);
}
REGISTER_PRIMITIVE_C(kNameCrop, Crop);
}  // namespace ops
}  // namespace mindspore
