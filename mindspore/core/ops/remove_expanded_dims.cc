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

#include "ops/remove_expanded_dims.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/core/ops/structure_ops.h"

namespace mindspore {
namespace ops {

static void RemNotExpandedDims(int64_t *idx_advanced, bool expand_true, int64_t tensor_index_ndim, int64_t rem_ndim,
                               std::vector<bool> *not_expanded_dim) {
  if (*idx_advanced != -1) {
    std::vector<bool> tensor_dims(tensor_index_ndim, true);
    if (expand_true) {
      tensor_dims = {false};
    }
    *idx_advanced = std::min(*idx_advanced, SizeToLong(not_expanded_dim->size()));
    (void)not_expanded_dim->insert(not_expanded_dim->begin() + *idx_advanced, tensor_dims.begin(), tensor_dims.end());
  }
  std::vector<bool> rem_ndim_vector(rem_ndim, true);
  (void)not_expanded_dim->insert(not_expanded_dim->end(), rem_ndim_vector.begin(), rem_ndim_vector.end());
  size_t count_leading_false = 0;
  while (count_leading_false < not_expanded_dim->size() && !((*not_expanded_dim)[count_leading_false])) {
    count_leading_false += 1;
  }
  *idx_advanced = std::max(static_cast<int64_t>(0), *idx_advanced - SizeToLong(count_leading_false));
}

static ShapeVector FilterExpandedDims(const ShapeVector &shape, const std::vector<bool> &not_expanded_dim) {
  int64_t diff = SizeToLong(not_expanded_dim.size()) - SizeToLong(shape.size());
  if (diff < 0) {
    MS_EXCEPTION(ValueError) << "Input array must have the same size across all dimensions.";
  }
  std::vector<int64_t> res;
  size_t index = std::min(shape.size(), not_expanded_dim.size() - static_cast<size_t>(diff));
  for (size_t i = 0; i < index; i++) {
    if (not_expanded_dim[(i + static_cast<size_t>(diff))]) {
      (void)res.emplace_back(shape[i]);
    }
  }
  return res;
}

std::tuple<int64_t, ShapeVector, int64_t> RemoveExpandedDims::ConstRemoveExpandedDims(
  bool has_true, bool has_false, bool has_sequence, const ShapeVector &broadcast_shape, int64_t rem_ndim,
  const ShapeVector &value_shape, const ShapeVector &data_shape, bool indices_out_empty, int64_t idx_advanced,
  std::vector<int64_t> new_tuple_index_types, size_t expand_dims_count) {
  int64_t indices_out = -1;
  ShapeVector reshape_info;
  size_t ellipse_position = 0;
  size_t not_ellipse_occupy_dims = 0;
  bool has_ellipsis = false;
  for (size_t i = 0; i < 8; i++) {
    if (new_tuple_index_types[i] == kMetaTypeEllipsis) {
      has_ellipsis = true;
      ellipse_position = i;
    } else if (new_tuple_index_types[i] != kTypeUnknown) {
      not_ellipse_occupy_dims += 1;
    }
  }
  size_t ellipse_occupy_dims = data_shape.size() + expand_dims_count - not_ellipse_occupy_dims;
  std::vector<bool> not_expanded_dim;
  for (size_t i = 0; i < new_tuple_index_types.size(); i++) {
    if (new_tuple_index_types[i] == kMetaTypeNone) {
      (void)not_expanded_dim.emplace_back(false);
    } else if (new_tuple_index_types[i] == kObjectTypeSlice) {
      (void)not_expanded_dim.emplace_back(true);
    } else if (has_ellipsis && i == ellipse_position) {
      std::vector<bool> empty_slice(ellipse_occupy_dims, true);
      (void)not_expanded_dim.insert(not_expanded_dim.end(), empty_slice.begin(), empty_slice.end());
    }
  }
  if (has_false) {
    if (std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<>()) != 1) {
      MS_EXCEPTION(IndexError) << "Unable to broadcast indices " << broadcast_shape;
    }
    indices_out = 0;
  } else {
    bool expand_true = has_true && !has_sequence;
    int64_t tensor_index_ndim = SizeToLong(broadcast_shape.size());
    RemNotExpandedDims(&idx_advanced, expand_true, tensor_index_ndim, rem_ndim, &not_expanded_dim);
    if (indices_out_empty) {
      indices_out = 1;
    }
    reshape_info = FilterExpandedDims(value_shape, not_expanded_dim);
  }
  return {indices_out, reshape_info, idx_advanced};
}

AbstractBasePtr RemoveExpandedDimsInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const size_t inputs_size = 5;
  CheckArgsSize(op_name, input_args, inputs_size);
  const AbstractBasePtr &data_abs = input_args[kIndex0];
  const AbstractBasePtr &value_abs = input_args[kIndex1];
  const AbstractBasePtr &has_false_abs = input_args[kIndex2];
  const AbstractBasePtr &broadcast_shape_abs = input_args[kIndex3];
  const AbstractBasePtr &idx_advanced_abs = input_args[kIndex4];

  ShapeVector value_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(value_abs->BuildShape())[kShape];
  ShapeVector data_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data_abs->BuildShape())[kShape];
  if (IsDynamic(value_shape) || IsDynamic(data_shape) || !IsValueKnown(has_false_abs->BuildValue()) ||
      !IsValueKnown(broadcast_shape_abs->BuildValue()) || idx_advanced_abs->isa<abstract::AbstractTensor>()) {
    auto abs_any = std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
    auto new_value_shape = std::vector<int64_t>{SizeToLong(value_shape.size())};
    if (IsDynamicRank(value_shape)) {
      new_value_shape = value_shape;
    }
    auto abs_tensor =
      std::make_shared<abstract::AbstractTensor>(abs_any, std::make_shared<abstract::Shape>(new_value_shape));
    auto scalar_abs_tensor = std::make_shared<abstract::AbstractTensor>(abs_any, std::make_shared<abstract::Shape>());

    AbstractBasePtrList abs_list{scalar_abs_tensor, abs_tensor, scalar_abs_tensor};
    return std::make_shared<abstract::AbstractTuple>(abs_list);
  }
  auto tensor_ptr = has_false_abs->BuildValue()->cast<mindspore::tensor::TensorPtr>();
  bool has_false = *reinterpret_cast<int64_t *>(tensor_ptr->data_c()) > 0;
  auto idx_advanced = GetValue<int64_t>(idx_advanced_abs->BuildValue());
  ShapeVector broadcast_shape = GetValue<std::vector<int64_t>>(broadcast_shape_abs->BuildValue());
  auto has_true = GetValue<bool>(primitive->GetAttr(kAttrHasTrue));
  auto has_sequence = GetValue<bool>(primitive->GetAttr(kAttrHasSequence));
  auto new_tuple_index_types = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrTupleIndexTypes));
  size_t valid_tensor_nums = 0;
  auto expand_dims = GetValue<int64_t>(primitive->GetAttr(kAttrExpandDimsCnt));
  for (size_t i = 0; i < new_tuple_index_types.size(); i++) {
    if (new_tuple_index_types[i] == kMetaTypeEllipsis) {
      valid_tensor_nums = data_shape.size() + expand_dims;
      break;
    } else if (new_tuple_index_types[i] != kTypeUnknown) {
      valid_tensor_nums += 1;
    }
  }
  auto rem_ndim = SizeToLong(data_shape.size()) - (SizeToLong(valid_tensor_nums) - expand_dims);
  bool empty_indices_out = GetValue<bool>(primitive->GetAttr(kAttrEmptyIndicesOut));
  auto [indices_out, new_value_shape, new_idx_advanced] = RemoveExpandedDims::ConstRemoveExpandedDims(
    has_true, has_false, has_sequence, broadcast_shape, rem_ndim, value_shape, data_shape, empty_indices_out,
    idx_advanced, new_tuple_index_types, static_cast<size_t>(expand_dims));
  abstract::AbstractBasePtrList elems;
  std::transform(new_value_shape.begin(), new_value_shape.end(), std::back_inserter(elems),
                 [](int64_t num) { return std::make_shared<abstract::AbstractScalar>(num); });

  auto indices_out_tensor = std::make_shared<tensor::Tensor>(indices_out);
  AbstractBasePtrList abs_list{indices_out_tensor->ToAbstract(), std::make_shared<abstract::AbstractTuple>(elems),
                               std::make_shared<abstract::AbstractScalar>(new_idx_advanced)};
  return std::make_shared<abstract::AbstractTuple>(abs_list);
}
MIND_API_OPERATOR_IMPL(RemoveExpandedDims, BaseOperator);
class RemoveExpandedDimsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RemoveExpandedDimsInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return RemoveExpandedDimsInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RemoveExpandedDimsInner(primitive, input_args);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(RemoveExpandedDims, prim::kPrimRemoveExpandedDims, RemoveExpandedDimsInfer, false);
}  // namespace ops
}  // namespace mindspore
