/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/conv2d_ext.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t expand_vec_size = 2;
constexpr size_t kInputArgsSize = 7;
constexpr size_t kConv2dExtInputDims = 4;
constexpr size_t kInputIdx = 0;
constexpr size_t kWightIdx = 1;
constexpr size_t kStrideIdx = 3;
constexpr size_t kPaddingIdx = 4;
constexpr size_t kDilationIdx = 5;
}  // namespace
BaseShapePtr Conv2DExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() != kInputArgsSize) {
    MS_LOG(EXCEPTION) << "input args size should be 5, but got " << input_args.size();
  }

  auto prim_name = primitive->name();
  auto input_shape_ptr = input_args[kInputIdx]->GetShape();
  auto weight_shape_ptr = input_args[kWightIdx]->GetShape();
  MS_EXCEPTION_IF_NULL(input_shape_ptr);
  MS_EXCEPTION_IF_NULL(weight_shape_ptr);
  const auto &input_shape = input_shape_ptr->GetShapeVector();
  const auto &weight_shape = weight_shape_ptr->GetShapeVector();

  if (input_shape.size() != kConv2dExtInputDims || weight_shape.size() != kConv2dExtInputDims) {
    MS_LOG(EXCEPTION) << "Input and weight shape size must be " << kConv2dExtInputDims
                      << ", but got input_shape:" << input_shape << ", weight_shape:" << weight_shape;
  }

  auto expande_dim_if_need = [input_args](const size_t &idx, const std::string &input_name) -> std::vector<int64_t> {
    auto value = input_args[idx]->BuildValue();
    auto vec = GetValue<std::vector<int64_t>>(value);

    if (vec.empty()) {
      MS_LOG(EXCEPTION) << "Input_name:" << input_name << " vec size is empty";
    }

    if (vec.size() == 1) {
      std::vector<int64_t> expand_vec;
      for (size_t i = 0; i < expand_vec_size; i++) {
        expand_vec.emplace_back(vec[0]);
      }
      return expand_vec;
    }

    return vec;
  };
  const auto &stride = expande_dim_if_need(kStrideIdx, "stride");
  const auto &padding = expande_dim_if_need(kPaddingIdx, "padding");
  const auto &dilation = expande_dim_if_need(kDilationIdx, "dilation");

  int64_t N = input_shape[0];
  int64_t H = input_shape[2];
  int64_t W = input_shape[3];

  auto kernel_size_0 = weight_shape[2];
  auto kernel_size_1 = weight_shape[3];

  int64_t Co = weight_shape[0];
  int64_t Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size_0 - 1) - 1) / stride[0] + 1;
  int64_t Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size_1 - 1) - 1) / stride[1] + 1;

  auto output_shape = {N, Co, Ho, Wo};
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr Conv2DExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  // TODO(wch) 1. bias dtype must be same with input， if bias is defined.

  const std::set<TypePtr> valid_types = {kInt8, kInt32, kInt64, kFloat16, kFloat32, kBFloat16};
  auto out_type =
    CheckAndConvertUtils::CheckTypeValid("input", input_args[kInputIdx]->GetType(), valid_types, primitive->name());

  return out_type;
}
}  // namespace ops
}  // namespace mindspore
