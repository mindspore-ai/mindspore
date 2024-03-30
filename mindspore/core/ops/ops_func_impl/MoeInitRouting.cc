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

#include "ops/ops_func_impl/minimum.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {

constexpr size_t kMatSize = 2;
constexpr size_t kInputX = 0;
constexpr size_t kInputRowIdx = 1;
constexpr size_t kInputExpertIdx = 2;
constexpr size_t kInputActiveNum = 3;
constexpr size_t koutputNum = 3;

BaseShapePtr MoeInitRoutingFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);

  auto prim_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputX]->GetShape());
  auto row_idx_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputRowIdx]->GetShape());
  auto expert_idx_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputExpertIdx]->GetShape());

  if (x_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'X' must be a Tensor type, but got:" << input_args[kInputX]->ToString();
  }
  if (row_idx_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'RowIdx' must be a Tensor type, but got:" << input_args[kInputRowIdx]->ToString();
  }
  if (expert_idx_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'ExpertIdx' must be a Tensor type, but got:" << input_args[kInputExpertIdx]->ToString();
  }

  ValuePtr active_num_ptr = input_args[kInputActiveNum]->GetValue();
  const int active_num = GetValue<int>(active_num);

  std::vector<abstract::BaseShapePtr> output_list;

  auto x_shp = x_shape_map[kShape];
  auto row_idx_shp = row_idx_shape_map[kShape];
  auto expert_idx_shp = expert_idx_shape_map[kShape];
  if (IsDynamicRank(x_shp) || IsDynamicRank(row_idx_shp) || IsDynamicRank(expert_idx_shp)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
    for (int64_t i = 0; i < koutputNum; ++i) {
      output_list.push_back(
        std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny}));
    }
    return std::make_shared<abstract::TupleShape>(std::move(output_list));
  }

  bool dynamic_shape = IsDynamic(x_shp) || IsDynamic(row_idx_shp) || IsDynamic(expert_idx_shp);
  if (!dynamic_shape) {
    CheckMoeInitRoutingInputSize(prim_name, "x", x_shp);
    CheckMoeInitRoutingInputSize(prim_name, "rowidx", row_idx_shp);
    CheckMoeInitRoutingInputSize(prim_name, "expertIdx", expert_idx_shp);
  }

  auto expanded_x_shape = input_shape;
  expanded_x_shape[0] = input_shape[0] * active_num;
  abstract::ShapePtr expanded_x_shape_ptr = std::make_shared<abstract::Shape>(expanded_x_shape);
  output_list.push_back(expanded_x_shape_ptr);

  ShapeVector expanded_row_idx_shape = {input_shape[0] * active_num}；
  abstract::ShapePtr expanded_row_idx_shape_ptr = std::make_shared<abstract::Shape>(expanded_row_idx_shape);
  output_list.push_back(expanded_row_idx_shape_ptr);

  ShapeVector expanded_expert_idx_shape = {input_shape[0] * active_num}；
  abstract::ShapePtr expanded_expert_idx_shape_ptr = std::make_shared<abstract::Shape>(expanded_expert_idx_shape);
  output_list.push_back(expanded_expert_idx_shape_ptr);

  return std::make_shared<abstract::TupleShape>(output_list);
}

TypePtr MoeInitRoutingFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = primitive->name();
  const auto &infer_type = input_args[kInputX]->GetType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_types, prim_name);

  const auto &row_idx_type = input_args[kInputRowIdx]->GetType();
  MS_EXCEPTION_IF_NULL(row_idx_type);
  const std::set<TypePtr> valid_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("row_idx", row_idx_type, valid_types, prim_name);

  const auto &expert_idx_type = input_args[kInputExpertIdx]->GetType();
  MS_EXCEPTION_IF_NULL(expert_idx_type);
  const std::set<TypePtr> valid_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("expert_idx", expert_idx_type, valid_types, prim_name);

  std::vector<TypePtr> type_tuple;
  type_tuple.push_back(infer_type->Clone());
  type_tuple.push_back(row_idx_type->Clone());
  type_tuple.push_back(expert_idx_type->Clone());

  return std::make_shared<Tuple>(std::move(type_tuple));
}
  
void MoeInitRoutingFuncImpl::CheckMoeInitRoutingInputSize(const std::string &op_name,
                                                               const std::string &input_name,
                                                               const ShapeVector &shape) const {
  constexpr size_t dim_limit = 2;
  if (shape.size() != dim_limit) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the input '" << input_name
                             << "' must be a 2D Tensor, but got " << shape.size() << "D shape "
                             << shape;
  }
}

}  // namespace ops
}  // namespace mindspore
