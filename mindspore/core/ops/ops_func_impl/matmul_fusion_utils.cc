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

#include "ops/ops_func_impl/matmul_fusion_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr MatmulFusionUtils::InferenceMultiMatmulInferShape(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
  auto shape_array_opt = GetArrayValue<int64_t>(input_args[kInputIndex2]);
  auto shape_array = shape_array_opt.value();
  std::vector<int64_t> shape_vec = shape_array.ToVector();

  if (IsDynamicRank(x_shape) || IsDynamicRank(w_shape)) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", dynamic rank is not supported";
  }
  const size_t x_rank = x_shape.size();
  const size_t w_rank = w_shape.size();
  MS_CHECK_VALUE(
    x_rank != 0 && x_rank == w_rank,
    CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', all inputs must have the same rank."));

  auto m = x_shape[0];
  auto k = x_shape[1];
  auto k0 = w_shape[1];
  MS_CHECK_VALUE(k == k0, CheckAndConvertUtils::FormatCommMsg(
                            "For '" + primitive->name() + "', the K axis of all inputs must have the same length."));

  MS_CHECK_VALUE(primitive->HasAttr("n_lens"),
                 CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', op must have attr 'n_lens'."));

  std::vector<int64_t> n_len_list = GetValue<std::vector<int64_t>>(primitive->GetAttr("n_lens"));
  MS_CHECK_VALUE(
    (n_len_list.size() == 2 || n_len_list.size() == 3),
    CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', attr 'n_lens' must have 2 or 3 value."));

  ShapeVector output_0_shape = {m, n_len_list[0]};
  ShapeVector output_1_shape = {m, n_len_list[1]};
  if (shape_vec.size() == 3) {
    output_0_shape = {shape_vec[0], shape_vec[1], n_len_list[0]};
    output_1_shape = {shape_vec[0], shape_vec[1], n_len_list[1]};
  }

  std::vector<BaseShapePtr> shape_lists;
  (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_0_shape));
  (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_1_shape));
  if (n_len_list.size() == 3) {
    ShapeVector output_2_shape = {m, n_len_list[2]};
    if (shape_vec.size() == 3) {
      output_2_shape = {shape_vec[0], shape_vec[1], n_len_list[2]};
    }
    (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_2_shape));
  }
  return std::make_shared<abstract::TupleShape>(shape_lists);
}

TuplePtr MatmulFusionUtils::InferenceMultiMatmulInferType(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_type = input_args[kInputIndex0]->GetType();
  if (x_type == kInt8 && input_args[kInputIndex1]->GetType() == kInt8) {
    x_type = kFloat16;
  }

  MS_CHECK_VALUE(primitive->HasAttr("n_lens"),
                 CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', op must have attr 'n_lens'."));
  std::vector<int64_t> n_len_list = GetValue<std::vector<int64_t>>(primitive->GetAttr("n_lens"));
  if (n_len_list.size() == 3) {
    return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, x_type, x_type});
  } else if (n_len_list.size() == 2) {
    return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, x_type});
  } else {
    MS_EXCEPTION(ValueError) << "n_lens's size must be 2 or 3 but got " << n_len_list.size();
  }
}

}  // namespace ops
}  // namespace mindspore
