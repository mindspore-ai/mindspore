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

#include "ops/stack.h"

namespace mindspore {
namespace ops {
namespace {
abstract::AbstractBasePtr StackInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);

  if (input_args.size() != 1) {
    MS_LOG(ERROR) << "Invalid output size:" << input_args.size();
  }
  if (input_args.size() < 1) {
    MS_LOG(ERROR) << "Invalid input size " << input_args.size();
  }
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  for (size_t i = 1; i < input_args.size(); ++i) {
    auto input_shape_tmp = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[i]->BuildShape())[kShape];
    if (input_shape_tmp.size() != input_shape.size()) {
      MS_LOG(ERROR) << "All input shape size should be the same!";
    }
    for (size_t j = 0; j < input_shape.size(); ++j) {
      if (input_shape_tmp.at(j) != input_shape.at(j)) {
        MS_LOG(ERROR) << "All input shape should be the same!";
      }
    }
  }
  std::vector<int64_t> infer_shape = input_shape;
  (void)infer_shape.insert(infer_shape.begin() + GetValue<int64_t>(primitive->GetAttr(kAxis)), input_args.size());

  auto infer_type0 = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  for (size_t i = 1; i < input_args.size(); i++) {
    if (input_args[i]->BuildType()->cast<TensorTypePtr>()->element() == infer_type0) {
      MS_LOG(ERROR) << "All input should have the same data type!input[" << i
                    << "] data type = " << input_args[i]->BuildType()->cast<TensorTypePtr>()->element();
    }
  }
  auto infer_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  auto output0 = std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape);
  AbstractBasePtrList output1 = {output0};
  return std::make_shared<abstract::AbstractTuple>(output1);
}
}  // namespace

void Stack::set_axis(const int64_t axis) { (void)AddAttr(kAxis, MakeValue(axis)); }

int64_t Stack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

void Stack::Init(const int64_t axis) { this->set_axis(axis); }

AbstractBasePtr StackInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(StackInfer(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameStack, Stack);
}  // namespace ops
}  // namespace mindspore
