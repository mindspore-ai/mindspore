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
  auto first_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto out_shape = first_input_shape;
  out_shape[out_shape.size() - 1] = GetValue<int64_t>(primitive->GetAttr(kFftLength)) / 2 + 1;
  out_shape.push_back(2);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return kComplex64;
}
}  // namespace

void Rfft::Init(const int64_t fft_length) { this->set_fft_length(fft_length); }

void Rfft::set_fft_length(const int64_t fft_length) { (void)this->AddAttr(kFftLength, MakeValue(fft_length)); }

int64_t Rfft::get_fft_length() const { return GetValue<int64_t>(GetAttr(kFftLength)); }

AbstractBasePtr RfftInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameRfft, Rfft);
}  // namespace ops
}  // namespace mindspore
