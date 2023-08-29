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

#include "ops/ops_func_impl/gather.h"
#include <memory>
#include <algorithm>
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
ShapeVector CalcuateGatherWithBatchDims(const PrimitivePtr &primitive, int64_t batch_dims, int64_t axis_val,
                                        const ShapeVector &ind_vec, const ShapeVector &params_vec) {
  MS_CHECK_VALUE(axis_val >= batch_dims, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                           "batch_dims", axis_val, kGreaterEqual, batch_dims, primitive));

  for (size_t i = 0; i < LongToSize(batch_dims); i++) {
    if (ind_vec[i] == abstract::TensorShape::kShapeDimAny || params_vec[i] != abstract::TensorShape::kShapeDimAny) {
      continue;
    }
    if (ind_vec[i] != params_vec[i]) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', params.shape[" << i
                        << "] should be equal to indices.shape[" << i << "] but got param.shape: " << params_vec
                        << ", indices.shape: " << ind_vec;
    }
  }

  ShapeVector out_vec;
  for (size_t i = 0; i < LongToSize(batch_dims); i++) {
    if (params_vec[i] != abstract::TensorShape::kShapeDimAny) {
      out_vec.push_back(params_vec[i]);
    } else {
      out_vec.push_back(ind_vec[i]);
    }
  }
  for (size_t i = LongToSize(batch_dims); i < LongToSize(axis_val); i++) {
    out_vec.push_back(params_vec[i]);
  }
  for (size_t i = LongToSize(batch_dims); i < ind_vec.size(); i++) {
    out_vec.push_back(ind_vec[i]);
  }
  for (size_t i = LongToSize(axis_val) + 1; i < params_vec.size(); i++) {
    out_vec.push_back(params_vec[i]);
  }
  return out_vec;
}

bool GetBatchDimByInput(std::vector<int64_t> *params_shape, std::vector<int64_t> *indices_shape, int64_t *batch_dims) {
  if (params_shape->front() == abstract::TensorShape::kShapeDimAny ||
      indices_shape->front() == abstract::TensorShape::kShapeDimAny) {
    return false;
  }

  if (params_shape->front() != indices_shape->front()) {
    *batch_dims = 0;
    return true;
  }
  return false;
}

BaseShapePtr GatherFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto params_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  auto indices_shape = input_args[kInputIndex1]->GetShape()->GetShapeVector();
  if (IsDynamicRank(params_shape) || IsDynamicRank(indices_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }

  constexpr size_t kGatherInputParamsMaxDim = 7;
  if (params_shape.size() > kGatherInputParamsMaxDim) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the dimension of 'input_params' should be "
                      << kGatherInputParamsMaxDim << "D or lower, but got " << params_shape.size() << ".";
  }

  int64_t params_shape_rank = SizeToLong(params_shape.size());
  int64_t indices_shape_rank = SizeToLong(indices_shape.size());

  auto batch_dims_value = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());
  int64_t batch_dims;
  if (!batch_dims_value.has_value()) {
    if (!GetBatchDimByInput(&params_shape, &indices_shape, &batch_dims)) {
      return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }
  } else {
    batch_dims = batch_dims_value.value();
  }

  MS_CHECK_VALUE(-params_shape_rank <= batch_dims && batch_dims < params_shape_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("batch_dims", batch_dims, kIncludeBoth,
                                                             {-params_shape_rank, params_shape_rank}, primitive));
  batch_dims = batch_dims < 0 ? batch_dims + params_shape_rank : batch_dims;

  auto axis_value = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  int64_t axis;
  if (!axis_value.has_value()) {
    std::vector<int64_t> dyn_output;
    int64_t output_rank = params_shape_rank + indices_shape_rank - batch_dims - 1;
    dyn_output.resize(output_rank, abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::TensorShape>(dyn_output);
  } else {
    axis = axis_value.value();
  }
  MS_CHECK_VALUE(-params_shape_rank <= axis && axis < params_shape_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis, kIncludeBoth,
                                                             {-params_shape_rank, params_shape_rank}, primitive));
  axis = axis < 0 ? axis + params_shape_rank : axis;

  if (batch_dims > 0) {
    auto output_shape = CalcuateGatherWithBatchDims(primitive, batch_dims, axis, indices_shape, params_shape);
    return std::make_shared<abstract::TensorShape>(output_shape);
  }

  auto calc_shape = [axis](const ShapeVector &ind_vec, const ShapeVector &params_vec) -> ShapeVector {
    ShapeVector out_vec;
    (void)std::copy(params_vec.begin(), params_vec.begin() + axis, std::back_inserter(out_vec));
    (void)copy(ind_vec.begin(), ind_vec.end(), std::back_inserter(out_vec));
    (void)copy(params_vec.begin() + axis + 1, params_vec.end(), std::back_inserter(out_vec));
    return out_vec;
  };
  auto output_shape = calc_shape(indices_shape, params_shape);
  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr GatherFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
