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

#include "ops/rfft.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto rfft_prim = primitive->cast<PrimRfftPtr>();
  MS_EXCEPTION_IF_NULL(rfft_prim);
  auto prim_name = rfft_prim->name();
  auto first_input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("first_input_shape", input_args[0]->BuildShape(), prim_name);
  auto out_shape = first_input_shape;
  out_shape[out_shape.size() - 1] = rfft_prim->get_fft_length() / 2 + 1;
  out_shape.push_back(2);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return TypeIdToType(kNumberTypeComplex64);
}
}  // namespace

void Rfft::Init(const int64_t fft_length) { this->set_fft_length(fft_length); }

void Rfft::set_fft_length(const int64_t fft_length) { this->AddAttr(kFftLength, MakeValue(fft_length)); }

int64_t Rfft::get_fft_length() const {
  auto value_ptr = this->GetAttr(kFftLength);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr RfftInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameRfft, Rfft);
}  // namespace ops
}  // namespace mindspore
