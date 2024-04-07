/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/convolution_grad.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kConvolutionGradInputArgsSize = 11;
constexpr size_t kConvolutionGradInputDims = 4;
}  // namespace
BaseShapePtr ConvolutionGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() != kConvolutionGradInputArgsSize) {
    MS_LOG(EXCEPTION) << "input args size should be " << kConvolutionGradInputArgsSize << ", but got "
                      << input_args.size();
  }

  auto x_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto weight_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto dout_shape_ptr = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(dout_shape_ptr);
  const auto &dout_shape = dout_shape_ptr->GetShapeVector();

  auto get_bias_grad_shape = [dout_shape]() {
    if (IsDynamicRank(dout_shape)) {
      return abstract::Shape::kShapeDimAny;
    }
    if (dout_shape.size() != kConvolutionGradInputDims) {
      MS_LOG(EXCEPTION) << "dout_shape size should be " << kConvolutionGradInputDims << ", but got "
                        << dout_shape.size();
    }
    return dout_shape[1];
  };

  ShapeVector bias_grad_shape = {get_bias_grad_shape()};
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    x_shape_ptr, weight_shape_ptr, std::make_shared<abstract::Shape>(bias_grad_shape)});
}

TypePtr ConvolutionGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type_ptr = input_args[kInputIndex1]->GetType();
  auto weight_type_ptr = input_args[kInputIndex2]->GetType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type_ptr, weight_type_ptr, x_type_ptr});
}
}  // namespace ops
}  // namespace mindspore
