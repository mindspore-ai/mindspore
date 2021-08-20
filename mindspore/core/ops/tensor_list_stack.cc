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

#include <vector>
#include <functional>

#include "ops/tensor_list_stack.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void TensorListStack::Init(const int64_t num_elements, const int64_t element_dtype) {
  this->set_num_elements(num_elements);
  this->set_element_dtype(element_dtype);
}

void TensorListStack::set_num_elements(const int64_t num_elements) {
  (void)this->AddAttr(kNumElements, MakeValue(num_elements));
}

void TensorListStack::set_element_dtype(const int64_t element_dtype) {
  (void)this->AddAttr(kElement_dtype, MakeValue(element_dtype));
}

int64_t TensorListStack::get_num_elements() const {
  auto value_ptr = GetAttr(kNumElements);
  return GetValue<int64_t>(value_ptr);
}

int64_t TensorListStack::get_element_dtype() const {
  auto value_ptr = GetAttr(kElement_dtype);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr TensorListStackInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  auto input0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  int64_t num = std::accumulate(input0_shape.begin(), input0_shape.end(), 1LL, std::multiplies<int64_t>());
  if (num == 0) {
    MS_LOG(ERROR) << "Try to stack a empty tensorlist!";
  }
  if (input_args[1]->BuildShape() == nullptr) {
    MS_LOG(ERROR) << "ele_shape->data_c() is nullptr";
  }
  auto input1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  (void)input1_shape.insert(input1_shape.begin(), 1);
  return std::make_shared<abstract::AbstractTensor>(input_args[0]->BuildType(), input1_shape);
}
REGISTER_PRIMITIVE_C(kNameTensorListStack, TensorListStack);
}  // namespace ops
}  // namespace mindspore
