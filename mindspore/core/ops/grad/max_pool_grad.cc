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

#include "ops/grad/max_pool_grad.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void MaxPoolGrad::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
                       const PadMode &pad_mode, const Format &data_format) {
  this->set_data_format(data_format);
  this->set_kernel_size(kernel_size);
  this->set_strides(strides);
  this->set_pad_mode(pad_mode);
}

void MaxPoolGrad::set_data_format(const Format &data_format) {
  int64_t swi = data_format;
  this->AddAttr(kFormat, MakeValue(swi));
}

Format MaxPoolGrad::get_data_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

void MaxPoolGrad::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  std::vector<int64_t> k_size = _grad_check_vector(kSize, kernel_size, this->name());
  k_size = this->get_data_format() == NCHW ? k_size : std::vector<int64_t>{k_size[0], k_size[2], k_size[3], k_size[1]};
  this->AddAttr(kSize, MakeValue(k_size));
}

void MaxPoolGrad::set_strides(const std::vector<int64_t> &strides) {
  std::vector<int64_t> stride_ = _grad_check_vector(kStrides, strides, this->name());
  stride_ =
    this->get_data_format() == NCHW ? stride_ : std::vector<int64_t>{stride_[0], stride_[2], stride_[3], stride_[1]};
  this->AddAttr(kStrides, MakeValue(stride_));
}

AbstractBasePtr MaxPoolGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]->BuildValue());
  auto x1_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x1_shape", input_args[0]->BuildShape(), op_name);
  auto tensor_type = input_args[0]->BuildType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto element = tensor_type->element();
  return std::make_shared<abstract::AbstractTensor>(element, x1_shape);
}

REGISTER_PRIMITIVE_EVAL_IMPL(MaxPoolGrad, prim::kPrimMaxPoolGrad, MaxPoolGradInfer);
REGISTER_PRIMITIVE_C(kNameMaxPoolGrad, MaxPoolGrad);
}  // namespace ops
}  // namespace mindspore
