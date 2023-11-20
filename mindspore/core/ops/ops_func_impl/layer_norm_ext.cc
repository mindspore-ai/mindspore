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
#include "ops/ops_func_impl/layer_norm_ext.h"
#include <functional>
#include <string>
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
constexpr int64_t kTwoD = 2;
constexpr int64_t kFourD = 4;
BaseShapePtr LayerNormExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  constexpr int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto input_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto norm_shape = GetValue<std::vector<int64_t>>(input_args[kInputIndex1]->BuildValue());
  const auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_shape_ptr)[kShape];
  const auto norm_dim = norm_shape.size();
  const auto input_dim = input_shape.size();
  const auto begin_axis = input_dim - norm_dim;
  if (input_dim < kTwoD || input_dim > kFourD) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', input_rank can expects 2d,3d,4d. But got: " << input_dim << "d.";
  }
  const int64_t m =
    std::accumulate(input_shape.cbegin(), input_shape.cbegin() + begin_axis, 1LL, std::multiplies<int64_t>());
  ShapeVector mean_out_shape, rstd_out_shape;
  if (m <= 0) {
    mean_out_shape = {m};
    rstd_out_shape = {m};
  } else {
    ShapeVector mean_shape;
    for (size_t i = 0; i < begin_axis; ++i) {
      (void)mean_shape.emplace_back(input_shape[i]);
    }
    for (size_t i = begin_axis; i < input_dim; ++i) {
      (void)mean_shape.emplace_back(1);
    }
    mean_out_shape = mean_shape;
    rstd_out_shape = mean_shape;
  }
  std::vector<BaseShapePtr> shapes_list = {input_shape_ptr};
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(mean_out_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(rstd_out_shape));
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr LayerNormExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  // outputs: output, mean_out, rstd_out
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  constexpr int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);

  auto input_type = input_args[kInputIndex0]->BuildType();
  auto weight_type = input_args[kInputIndex2]->BuildType();
  auto bias_type = input_args[kInputIndex3]->BuildType();
  auto valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_type", input_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("weight_type", weight_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("bias_type", bias_type, valid_types, op_name);

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::vector<TypePtr> types_list;
  types_list = {input_type, input_type, input_type};
  return std::make_shared<Tuple>(types_list);
}
}  // namespace ops
}  // namespace mindspore
